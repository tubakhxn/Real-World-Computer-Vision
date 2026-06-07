#!/usr/bin/env python3
"""
PORT BULK CARRIER INTELLIGENCE  v6.0
Dev: tubakhxn
- NEW: Colored segmentation overlay — Ship (MAGENTA), Cranes (LIME GREEN)
- NEW: Water masking — sea/ocean pixels NOT colored
- NEW: Smooth filled polygon segmentation (like SAM-style coloring)
- Faster: detect_every=4, scale=0.45
- Better HUD with real-time progress bar
"""

import subprocess, sys, os, time, warnings, glob, threading, math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

warnings.filterwarnings("ignore")

_REQUIRED = ["opencv-python", "numpy", "matplotlib", "tqdm"]

def _ensure():
    for pkg in _REQUIRED:
        mod = pkg.replace("-", "_").split("[")[0]
        try:
            __import__(mod)
        except ImportError:
            print(f"  [AUTO-INSTALL] {pkg} ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

print("=" * 70)
print("  PORT BULK CARRIER INTELLIGENCE  |  tubakhxn  |  v6.0")
print("  Segmentation: Ship=MAGENTA  Cranes=LIME  Water=NO COLOR")
print("=" * 70)
_ensure()

import cv2
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

CFG = {
    "output_video":       "port_output_v6.mp4",
    "output_dashboard":   "port_dashboard_v6.png",
    "output_dir":         "port_output_v6",
    "detect_every":       4,
    "detect_scale":       0.45,
    "max_detect_dim":     640,
    "min_crane_area":     700,
    "min_ship_area":      6000,
    "iou_match_thresh":   0.18,
    "max_age_frames":     12,
    "trail_length":       35,
    "hud_width":          310,
    "heatmap_decay":      0.983,
    # Segmentation overlay alphas
    "seg_alpha_ship":     0.52,
    "seg_alpha_crane":    0.48,
}

# Colors in BGR
C = {
    "crane":   (50,  255,  50),      # LIME GREEN
    "ship":    (200,  50, 200),      # MAGENTA / PURPLE-PINK
    "accent":  (0,   210, 255),      # CYAN
    "bg":      (8,    10,  16),
    "panel":   (12,   15,  24),
    "grey":    (80,   85, 100),
    "white":   (235, 238, 245),
    "dark":    (20,   24,  36),
    "green":   (30,  220,  80),
    "amber":   (0,   165, 255),
}

# Segmentation fill colors (BGR)
SEG_SHIP_COLOR  = (180, 40, 190)    # magenta
SEG_CRANE_COLOR = (40,  240, 40)    # lime green


@dataclass
class BBox:
    x1: int; y1: int; x2: int; y2: int

    @property
    def cx(self) -> int:   return (self.x1 + self.x2) // 2
    @property
    def cy(self) -> int:   return (self.y1 + self.y2) // 2
    @property
    def w(self) -> int:    return self.x2 - self.x1
    @property
    def h(self) -> int:    return self.y2 - self.y1
    @property
    def area(self) -> float: return float(self.w * self.h)
    @property
    def center(self) -> Tuple[int, int]: return (self.cx, self.cy)

    def iou(self, o: "BBox") -> float:
        ix1 = max(self.x1, o.x1); iy1 = max(self.y1, o.y1)
        ix2 = min(self.x2, o.x2); iy2 = min(self.y2, o.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return inter / (self.area + o.area - inter + 1e-6)


@dataclass
class Det:
    cls: str; bbox: BBox; conf: float; mask: Optional[np.ndarray] = None


@dataclass
class Track:
    tid: int; cls: str; bbox: BBox; conf: float
    mask: Optional[np.ndarray] = None
    trail: deque = field(default_factory=lambda: deque(maxlen=CFG["trail_length"]))
    missed: int = 0; age: int = 0
    velocity: Tuple[float, float] = (0., 0.)
    _pcx: float = 0.; _pcy: float = 0.

    def update(self, bbox: BBox, conf: float, mask=None):
        dx = bbox.cx - self._pcx; dy = bbox.cy - self._pcy
        a = 0.35
        self.velocity = (a * dx + (1 - a) * self.velocity[0],
                         a * dy + (1 - a) * self.velocity[1])
        self._pcx = float(bbox.cx); self._pcy = float(bbox.cy)
        self.bbox = bbox; self.conf = conf; self.mask = mask
        self.trail.append((bbox.cx, bbox.cy))
        self.missed = 0; self.age += 1


# ──────────────────────────────────────────────────────────────────────
#  WATER MASK  – exclude sea/ocean from coloring
# ──────────────────────────────────────────────────────────────────────
def build_water_mask(frame_small: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask (uint8, 255=water) at the small frame resolution.
    Detects teal/blue-green water using HSV.
    """
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    # Teal-blue-green water range
    m1 = cv2.inRange(hsv, (85,  40, 60), (105, 220, 220))   # teal
    m2 = cv2.inRange(hsv, (78,  30, 50), (115, 255, 210))   # broader blue-green
    water = cv2.bitwise_or(m1, m2)
    # Morphological cleaning
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, k, iterations=3)
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN,  k, iterations=2)
    return water


# ──────────────────────────────────────────────────────────────────────
#  DETECTOR
# ──────────────────────────────────────────────────────────────────────
class BulkPortDetector:
    def __init__(self, full_W: int, full_H: int):
        scale = min(CFG["detect_scale"], CFG["max_detect_dim"] / max(full_W, full_H))
        self.scale = scale
        self.dW = int(full_W * scale); self.dH = int(full_H * scale)
        self.inv = 1. / scale
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=36, detectShadows=False)
        print(f"  [DETECT] {full_W}x{full_H} -> {self.dW}x{self.dH}  scale={scale:.2f}")

    def _scale_area(self, a: int) -> int:
        return max(1, int(a * self.scale * self.scale))

    @staticmethod
    def _blobs(mask, min_area, max_n=12):
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = sorted(range(1, nb),
                       key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        out = []
        for i in blobs[:max_n]:
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out.append((stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                             stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                             labels == i))
        return out

    def _clean(self, mask, min_area, dil=2, ero=1):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if ero:  mask = cv2.erode(mask,  k, iterations=ero)
        if dil:  mask = cv2.dilate(mask, k, iterations=dil)
        nb, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, nb):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    def detect(self, frame: np.ndarray) -> List[Det]:
        small = cv2.resize(frame, (self.dW, self.dH), interpolation=cv2.INTER_LINEAR)
        hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        inv   = self.inv
        dets  = []

        # ── WATER MASK (do NOT color water) ──────────────────────────
        water_mask = build_water_mask(small)
        not_water  = cv2.bitwise_not(water_mask)

        # ── CRANE: white/light-grey + yellow machinery + edge density ─
        m_white  = cv2.inRange(hsv, (0,  0, 175), (180, 40, 255))
        m_lgrey  = cv2.inRange(hsv, (0,  0, 145), (180, 35, 215))
        m_yellow = cv2.inRange(hsv, (18, 90, 110), (35, 255, 255))
        crane_color = cv2.bitwise_or(cv2.bitwise_or(m_white, m_lgrey), m_yellow)
        crane_color = cv2.bitwise_and(crane_color, not_water)
        edges    = cv2.Canny(gray, 45, 130)
        edge_dil = cv2.dilate(edges, np.ones((9, 9), np.uint8))
        crane_mask = cv2.bitwise_and(crane_color, edge_dil)
        s_crane  = self._scale_area(CFG["min_crane_area"])
        crane_mask = self._clean(crane_mask, s_crane, dil=3, ero=1)

        for x1, y1, w_, h_, blob_lbl in self._blobs(crane_mask, s_crane, max_n=10):
            ar = max(w_, h_) / max(min(w_, h_), 1)
            if ar < 1.2 and w_ * h_ < s_crane * 3: continue
            fx1 = int(x1 * inv); fy1 = int(y1 * inv)
            fx2 = int((x1 + w_) * inv); fy2 = int((y1 + h_) * inv)
            # Upscale mask to full frame resolution
            seg = cv2.resize(blob_lbl.astype(np.uint8) * 255,
                             (frame.shape[1], frame.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
            dets.append(Det(cls="Crane", bbox=BBox(fx1, fy1, fx2, fy2),
                            conf=0.85, mask=seg))

        # ── SHIP BODY: large, bright-colored or grey vessel hull ──────
        # Ships appear as large regions that are NOT water and NOT crane.
        # Use color + large area approach.
        m_hull_grey  = cv2.inRange(hsv, (0,   0,  80), (180,  50, 200))
        m_hull_color = cv2.inRange(hsv, (0,  30,  60), (180, 255, 240))
        fg = self._bg.apply(small, learningRate=0.005)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg_wide = cv2.dilate(fg, np.ones((20, 20), np.uint8))

        # Ship = hull color AND not water, either fg or solid presence
        ship_cand = cv2.bitwise_or(m_hull_grey, m_hull_color)
        ship_cand = cv2.bitwise_and(ship_cand, not_water)
        # Exclude very bright (sky/white crane parts already handled)
        m_vbright = cv2.inRange(hsv, (0, 0, 240), (180, 20, 255))
        ship_cand = cv2.bitwise_and(ship_cand, cv2.bitwise_not(m_vbright))

        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        ship_cand = cv2.morphologyEx(ship_cand, cv2.MORPH_CLOSE, close_k)
        s_ship = self._scale_area(CFG["min_ship_area"])
        ship_cand = self._clean(ship_cand, s_ship, dil=5, ero=2)

        for x1, y1, w_, h_, blob_lbl in self._blobs(ship_cand, s_ship, max_n=4):
            ar = max(w_, h_) / max(min(w_, h_), 1)
            if ar > 8: continue
            if w_ * h_ < s_ship: continue
            fx1 = int(x1 * inv); fy1 = int(y1 * inv)
            fx2 = int((x1 + w_) * inv); fy2 = int((y1 + h_) * inv)
            seg = cv2.resize(blob_lbl.astype(np.uint8) * 255,
                             (frame.shape[1], frame.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
            dets.append(Det(cls="Ship", bbox=BBox(fx1, fy1, fx2, fy2),
                            conf=0.82, mask=seg))

        return self._nms(dets)

    @staticmethod
    def _nms(dets, thresh=0.45):
        keep = []; groups = defaultdict(list)
        for d in dets: groups[d.cls].append(d)
        for cls, grp in groups.items():
            grp.sort(key=lambda d: d.conf, reverse=True)
            sup = set()
            for i, d in enumerate(grp):
                if i in sup: continue
                keep.append(d)
                for j in range(i + 1, len(grp)):
                    if grp[i].bbox.iou(grp[j].bbox) > thresh: sup.add(j)
        return keep


# ──────────────────────────────────────────────────────────────────────
#  TRACKER
# ──────────────────────────────────────────────────────────────────────
class Tracker:
    def __init__(self):
        self.tracks = {}; self._nid = 0; self._retired = 0

    def update(self, dets):
        matched = set()
        for tid in list(self.tracks.keys()):
            tr = self.tracks[tid]; best_s, best_di = 0., None
            for di, d in enumerate(dets):
                if d.cls != tr.cls: continue
                s = tr.bbox.iou(d.bbox)
                if s > best_s: best_s, best_di = s, di
            if best_s >= CFG["iou_match_thresh"] and best_di is not None:
                d = dets[best_di]
                tr.update(d.bbox, d.conf, d.mask)
                matched.add(best_di)
            else:
                tr.missed += 1; tr.trail.append(None)
                if tr.missed > CFG["max_age_frames"]:
                    self._retired += 1; del self.tracks[tid]
        for di, d in enumerate(dets):
            if di not in matched:
                t = Track(tid=self._nid, cls=d.cls, bbox=d.bbox,
                          conf=d.conf, mask=d.mask)
                t._pcx = float(d.bbox.cx); t._pcy = float(d.bbox.cy)
                t.trail.append((d.bbox.cx, d.bbox.cy))
                self.tracks[self._nid] = t; self._nid += 1
        return self.tracks

    @property
    def total_processed(self): return self._retired


# ──────────────────────────────────────────────────────────────────────
#  HEATMAP
# ──────────────────────────────────────────────────────────────────────
class Heatmap:
    def __init__(self): self._m = None; self._cum = None

    def update(self, tracks, shape):
        h, w = shape[:2]
        if self._m is None:
            self._m   = np.zeros((h, w), np.float32)
            self._cum = np.zeros((h, w), np.float32)
        self._m *= CFG["heatmap_decay"]
        for tr in tracks.values():
            r = 70 if tr.cls == "Ship" else 30
            cx, cy = tr.bbox.cx, tr.bbox.cy
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(self._m,   (cx, cy), r, 1., -1)
                cv2.circle(self._cum, (cx, cy), r, 1., -1)

    def overlay(self, frame, alpha=0.14):
        if self._m is None: return
        norm = cv2.normalize(self._m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        col  = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
        mask = (norm > 20).astype(np.float32) / 255. * alpha
        for c in range(3):
            frame[:, :, c] = np.clip(
                frame[:, :, c] * (1 - mask) + col[:, :, c] * mask, 0, 255
            ).astype(np.uint8)

    @property
    def data(self): return self._cum


# ──────────────────────────────────────────────────────────────────────
#  ASYNC WRITER
# ──────────────────────────────────────────────────────────────────────
class AsyncWriter:
    def __init__(self, w):
        self._w = w; self._q = deque(); self._lock = threading.Lock()
        self._done = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def write(self, f):
        with self._lock: self._q.append(f)

    def _run(self):
        while not self._done or self._q:
            if self._q:
                with self._lock: f = self._q.popleft()
                self._w.write(f)
            else: time.sleep(0.001)

    def close(self): self._done = True; self._t.join(); self._w.release()


# ──────────────────────────────────────────────────────────────────────
#  SEGMENTATION OVERLAY  (the colored fill like the reference image)
# ──────────────────────────────────────────────────────────────────────
def apply_segmentation_overlay(frame: np.ndarray, tracks: dict) -> np.ndarray:
    """
    Draw filled colored masks:
      - Ship  → MAGENTA
      - Crane → LIME GREEN
    Water pixels are NOT colored (mask already excludes water).
    """
    out = frame.copy()
    H, W = frame.shape[:2]

    # Draw ships first (background layer), then cranes on top
    for cls_order in ["Ship", "Crane"]:
        for tr in tracks.values():
            if tr.cls != cls_order: continue
            if tr.mask is None:
                # Fallback: fill bounding box
                b = tr.bbox
                x1 = max(0, b.x1); y1 = max(0, b.y1)
                x2 = min(W, b.x2); y2 = min(H, b.y2)
                roi = out[y1:y2, x1:x2]
                color = SEG_SHIP_COLOR if tr.cls == "Ship" else SEG_CRANE_COLOR
                alpha = CFG["seg_alpha_ship"] if tr.cls == "Ship" else CFG["seg_alpha_crane"]
                overlay_roi = np.full_like(roi, color, dtype=np.uint8)
                out[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, overlay_roi, alpha, 0)
                continue

            # Use the actual pixel mask
            seg = tr.mask
            if seg.shape[:2] != (H, W):
                seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
            where = seg > 128
            color = SEG_SHIP_COLOR if tr.cls == "Ship" else SEG_CRANE_COLOR
            alpha = CFG["seg_alpha_ship"] if tr.cls == "Ship" else CFG["seg_alpha_crane"]
            for c_idx, c_val in enumerate(color):
                out[:, :, c_idx] = np.where(
                    where,
                    np.clip(frame[:, :, c_idx] * (1 - alpha) + c_val * alpha, 0, 255).astype(np.uint8),
                    out[:, :, c_idx]
                )

    return out


# ──────────────────────────────────────────────────────────────────────
#  RENDERER
# ──────────────────────────────────────────────────────────────────────
class Renderer:
    def __init__(self, W, H, hw):
        self.W = W; self.H = H; self.hw = hw

    @staticmethod
    def _T(img, text, x, y, sc=0.50, col=(220, 225, 235), bold=False, shadow=True):
        th = 2 if bold else 1
        if shadow:
            cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                        sc, (0, 0, 0), th + 1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)

    def _draw_trails(self, frame, tracks):
        for tr in tracks.values():
            pts = [p for p in tr.trail if p is not None]
            if len(pts) < 2: continue
            col = C["crane"] if tr.cls == "Crane" else C["ship"]
            for i in range(1, len(pts)):
                a = i / len(pts)
                cv2.line(frame, pts[i - 1], pts[i],
                         tuple(int(v * a) for v in col), 1, cv2.LINE_AA)

    def _draw_boxes(self, frame, tracks):
        for tr in tracks.values():
            b = tr.bbox
            col = C["crane"] if tr.cls == "Crane" else C["ship"]
            # Thin border only (fill already done by segmentation)
            cv2.rectangle(frame, (b.x1, b.y1), (b.x2, b.y2), col, 2, cv2.LINE_AA)
            # Corner tick marks
            tl = min(18, b.w // 4, b.h // 4)
            for sx, sy, dx, dy in [(b.x1, b.y1, 1, 1), (b.x2, b.y1, -1, 1),
                                    (b.x1, b.y2, 1, -1), (b.x2, b.y2, -1, -1)]:
                cv2.line(frame, (sx, sy), (sx + dx * tl, sy), col, 2, cv2.LINE_AA)
                cv2.line(frame, (sx, sy), (sx, sy + dy * tl), col, 2, cv2.LINE_AA)
            lbl = f"{tr.cls} #{tr.tid}  {tr.conf:.0%}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(frame, (b.x1, b.y1 - th - 10), (b.x1 + tw + 8, b.y1), col, -1)
            cv2.putText(frame, lbl, (b.x1 + 4, b.y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_hud(self, panel, tracks, tracker, fi, total):
        panel[:] = C["bg"]; hw = self.hw
        cv2.line(panel, (0, 0), (0, self.H), C["accent"], 3)

        def T(t, x, y, sc=0.42, col=C["white"], bold=False):
            self._T(panel, t, x, y, sc, col, bold, shadow=False)

        # Header
        cv2.rectangle(panel, (0, 0), (hw, 60), C["panel"], -1)
        T("PORT BULK CARRIER", 8, 26, 0.56, C["accent"], True)
        T("tubakhxn  |  v6.0  |  SEG EDITION", 8, 46, 0.30, C["grey"])
        cv2.line(panel, (4, 60), (hw - 4, 60), C["accent"], 1)

        n_crane = sum(1 for tr in tracks.values() if tr.cls == "Crane")
        n_ship  = sum(1 for tr in tracks.values() if tr.cls == "Ship")

        kpis = [
            ("CRANES ACTIVE", str(n_crane), C["crane"]),
            ("SHIPS VISIBLE",  str(n_ship),  C["ship"]),
            ("TOTAL TRACKED", str(len(tracks)), C["accent"]),
            ("RETIRED",       str(tracker.total_processed), C["green"]),
        ]
        cw = (hw - 14) // 2; ky = 70
        for i, (lbl, val, col) in enumerate(kpis):
            ox = 7 + (i % 2) * (cw); oy = ky + (i // 2) * 58
            cv2.rectangle(panel, (ox, oy), (ox + cw - 2, oy + 52), C["dark"], -1)
            cv2.rectangle(panel, (ox, oy), (ox + cw - 2, oy + 3), col, -1)
            T(lbl[:14], ox + 5, oy + 20, 0.28, C["grey"])
            T(val,      ox + 5, oy + 44, 0.70, col, True)

        y = ky + 2 * 58 + 12
        cv2.line(panel, (4, y), (hw - 4, y), (28, 32, 48), 1); y += 10

        # Crane movement section
        T("CRANE MOVEMENT", 8, y + 14, 0.42, C["accent"]); y += 22
        shown = 0
        for tr in [t for t in tracks.values() if t.cls == "Crane"][:5]:
            spd = math.hypot(tr.velocity[0], tr.velocity[1])
            mov = "ACTIVE" if spd > 0.5 else "STATIC"
            T(f"  #{tr.tid}: {mov} ({spd:.1f}px/f)", 8, y + 14,
              0.36, C["green"] if spd > 0.5 else C["grey"])
            y += 18; shown += 1
        if shown == 0:
            T("  None detected", 8, y + 14, 0.36, C["grey"]); y += 18

        cv2.line(panel, (4, y + 4), (hw - 4, y + 4), (28, 32, 48), 1); y += 14

        # Ship section
        T("SHIP STATUS", 8, y + 14, 0.42, C["accent"]); y += 22
        shown = 0
        for tr in [t for t in tracks.values() if t.cls == "Ship"][:3]:
            T(f"  #{tr.tid}: {tr.bbox.area / 1000.:.0f}k px area",
              8, y + 14, 0.36, C["ship"])
            y += 18; shown += 1
        if shown == 0:
            T("  None detected", 8, y + 14, 0.36, C["grey"]); y += 18

        cv2.line(panel, (4, y + 4), (hw - 4, y + 4), (28, 32, 48), 1); y += 14

        # Loading status
        T("LOADING STATUS", 8, y + 14, 0.42, C["accent"]); y += 22
        if n_crane > 0 and n_ship > 0:   status, sc = "LOADING IN PROGRESS", C["green"]
        elif n_crane > 0:                 status, sc = "CRANES STANDBY",       C["amber"]
        elif n_ship > 0:                  status, sc = "SHIP DOCKED",          C["ship"]
        else:                             status, sc = "SCANNING...",           C["grey"]
        cv2.rectangle(panel, (8, y), (hw - 8, y + 28), C["dark"], -1)
        cv2.rectangle(panel, (8, y), (hw - 8, y + 28), sc, 2)
        (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        T(status, hw // 2 - tw // 2, y + 20, 0.44, sc, True); y += 38

        cv2.line(panel, (4, y + 4), (hw - 4, y + 4), (28, 32, 48), 1); y += 14

        # Legend with colored squares
        T("LEGEND", 8, y + 14, 0.42, C["accent"]); y += 22
        cv2.rectangle(panel, (10, y), (30, y + 14), C["crane"], -1)
        T("Ship Crane / Boom", 34, y + 13, 0.37, C["crane"]); y += 22
        cv2.rectangle(panel, (10, y), (30, y + 14), C["ship"], -1)
        T("Container Ship",   34, y + 13, 0.37, C["ship"]); y += 22
        cv2.rectangle(panel, (10, y), (30, y + 14), (100, 80, 60), -1)
        T("Water (no color)", 34, y + 13, 0.37, C["grey"]); y += 22

        # Progress bar at bottom
        prog = fi / max(1, total)
        bar_y = self.H - 30
        cv2.rectangle(panel, (4, bar_y), (hw - 4, bar_y + 14), C["dark"], -1)
        bar_fill = int(4 + (hw - 8) * prog)
        cv2.rectangle(panel, (4, bar_y), (bar_fill, bar_y + 14), C["accent"], -1)
        T(f"Frame {fi}/{total}  {prog * 100:.1f}%", 8, self.H - 4, 0.32, C["grey"])

    def render(self, frame, tracks, heatmap, tracker, fi, total, is_kf):
        H, W = frame.shape[:2]; hw = self.hw
        canvas = np.zeros((H, W + hw, 3), np.uint8)

        # 1. Apply segmentation colored overlay to main frame area
        seg_frame = apply_segmentation_overlay(frame, tracks)
        canvas[:, :W] = seg_frame
        main = canvas[:, :W]

        # 2. Subtle heatmap on top
        heatmap.overlay(main, alpha=0.12)

        # 3. Trails and boxes
        if is_kf: self._draw_trails(main, tracks)
        self._draw_boxes(main, tracks)

        # 4. Mini counter overlay (top-left)
        n_crane = sum(1 for t in tracks.values() if t.cls == "Crane")
        n_ship  = sum(1 for t in tracks.values() if t.cls == "Ship")
        lines   = [(f"Cranes : {n_crane}", C["crane"]),
                   (f"Ships  : {n_ship}",  C["ship"])]
        bh = len(lines) * 28 + 12
        cv2.rectangle(main, (0, 0), (240, bh), (0, 0, 0), -1)
        cv2.line(main, (0, bh), (240, bh), C["accent"], 2)
        yt = 26
        for txt, col in lines:
            self._T(main, txt, 10, yt, 0.54, col, bold=True, shadow=True)
            yt += 28

        # 5. Separator line + HUD panel
        cv2.line(canvas, (W, 0), (W, H), C["accent"], 2)
        self._draw_hud(canvas[:, W:], tracks, tracker, fi, total)
        return canvas


# ──────────────────────────────────────────────────────────────────────
#  DASHBOARD
# ──────────────────────────────────────────────────────────────────────
def generate_dashboard(history, heatmap_data, out):
    print("\n[DASHBOARD] Generating ...")
    F = list(range(len(history)))
    BG   = "#07090f"; PAN = "#0d1018"; ACC = "#00c8ff"
    MGT  = "#c832c8"; LGN = "#32f032"; TXT = "#c8d0dc"; GRID = "#1a1e2a"

    fig = plt.figure(figsize=(20, 10), facecolor=BG)
    fig.suptitle(
        "PORT BULK CARRIER INTELLIGENCE v6.0  —  OPERATIONS DASHBOARD",
        color=ACC, fontsize=16, fontweight="bold", y=0.98, fontfamily="monospace"
    )
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.38,
                          left=0.06, right=0.97, top=0.93, bottom=0.06)

    def sax(ax, title=""):
        ax.set_facecolor(PAN)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.tick_params(colors=TXT, labelsize=8)
        if title: ax.set_title(title, color=ACC, fontsize=10,
                               fontfamily="monospace", pad=6)

    crane_t = [h["n_crane"] for h in history]
    ship_t  = [h["n_ship"]  for h in history]

    ax = fig.add_subplot(gs[0, :2]); sax(ax, "Detections Over Time")
    if F:
        ax.fill_between(F, crane_t, alpha=.28, color=LGN, label="Cranes")
        ax.fill_between(F, ship_t,  alpha=.28, color=MGT, label="Ships")
        ax.plot(F, crane_t, color=LGN, lw=1.8)
        ax.plot(F, ship_t,  color=MGT, lw=1.8)
    ax.legend(facecolor=PAN, edgecolor=GRID, labelcolor=TXT, fontsize=9)
    ax.set_xlabel("Frame", color=TXT, fontsize=8)
    ax.yaxis.grid(True, color=GRID, alpha=.5)

    ax = fig.add_subplot(gs[0, 2]); sax(ax, "Activity Heatmap")
    if heatmap_data is not None and heatmap_data.max() > 0:
        ax.imshow(cv2.resize(heatmap_data, (480, 270)), cmap="hot",
                  interpolation="bilinear", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.text(.5, .5, "No data", color=TXT, ha="center", va="center",
                transform=ax.transAxes)

    ax = fig.add_subplot(gs[1, :2]); sax(ax, "Object Count Distribution")
    total_t = [h["n_total"] for h in history]
    if total_t:
        n, bins, patches = ax.hist(total_t, bins=20, edgecolor=BG, alpha=.85)
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor(LGN if left < 4 else MGT)
    ax.set_xlabel("Objects", color=TXT, fontsize=8)
    ax.yaxis.grid(True, color=GRID, alpha=.5)

    ax = fig.add_subplot(gs[1, 2]); sax(ax, "Detection Split")
    totc = sum(crane_t); tots = sum(ship_t)
    if totc + tots > 0:
        ax.pie([totc, tots], labels=["Cranes", "Ships"],
               colors=[LGN, MGT], autopct="%1.0f%%", startangle=90,
               textprops={"color": TXT, "fontsize": 9},
               wedgeprops={"edgecolor": BG, "linewidth": 2})
    else:
        ax.text(.5, .5, "No data", color=TXT, ha="center", va="center",
                transform=ax.transAxes)

    plt.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"  [OK] Dashboard -> {out}")


# ──────────────────────────────────────────────────────────────────────
#  VIDEO WRITER
# ──────────────────────────────────────────────────────────────────────
def get_writer(path, fps, size):
    for fc in ["H264", "avc1", "h264"]:
        try:
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fc), fps, size)
            if w.isOpened():
                print(f"  [CODEC] {fc} -> {path}"); return w, path
            w.release()
        except: pass
    avi_path = path.replace(".mp4", ".avi")
    try:
        w = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"XVID"), fps, size)
        if w.isOpened():
            print(f"  [CODEC] XVID -> {avi_path}  (open with VLC if WMP fails)")
            return w, avi_path
        w.release()
    except: pass
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    print(f"  [CODEC] mp4v -> {path}  (open with VLC)")
    return w, path


# ──────────────────────────────────────────────────────────────────────
#  FIND VIDEO
# ──────────────────────────────────────────────────────────────────────
def find_video() -> str:
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]): return sys.argv[1]
    for pat in ["port*.mp4", "harbor*.mp4", "ship*.mp4", "bulk*.mp4",
                "*.mp4", "*.avi", "*.mov", "*.mkv"]:
        for f in glob.glob(pat):
            if "output" not in f.lower() and "dashboard" not in f.lower() \
               and os.path.isfile(f):
                return f
    return ""


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────
def run():
    import shutil
    inp = find_video()
    if not inp:
        print("\n[ERROR] No video found.")
        print("  Usage: python port_bulk_carrier_v6.py your_video.mp4")
        sys.exit(1)
    print(f"\n[VIDEO] Input -> {inp}")

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {inp}"); sys.exit(1)

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO] {W}x{H}  {fps:.1f}fps  {total} frames  ({total/fps:.1f}s)")

    os.makedirs(CFG["output_dir"], exist_ok=True)
    hw       = CFG["hud_width"]
    out_path = os.path.join(CFG["output_dir"], CFG["output_video"])
    raw_w, actual_path = get_writer(out_path, fps, (W + hw, H))
    writer   = AsyncWriter(raw_w)

    detector = BulkPortDetector(W, H)
    tracker  = Tracker()
    heatmap  = Heatmap()
    renderer = Renderer(W, H, hw)

    history      = []
    detect_every = CFG["detect_every"]
    last_tracks  = {}

    print(f"\n[PROCESS] Analysing {total} frames ...")
    print(f"  detect_every={detect_every}  scale={CFG['detect_scale']}")
    print(f"  Segmentation: Ship=MAGENTA  Crane=LIME  Water=EXCLUDED")
    t0 = time.time()

    for fi in tqdm(range(total), desc="  Frames", ncols=72, unit="fr"):
        ret, frame = cap.read()
        if not ret: break
        is_kf = (fi % detect_every == 0)
        if is_kf:
            dets        = detector.detect(frame)
            last_tracks = tracker.update(dets)
            heatmap.update(last_tracks, frame.shape)
        canvas = renderer.render(frame, last_tracks, heatmap,
                                 tracker, fi, total, is_kf)
        writer.write(canvas)
        if is_kf:
            n_crane = sum(1 for t in last_tracks.values() if t.cls == "Crane")
            n_ship  = sum(1 for t in last_tracks.values() if t.cls == "Ship")
            history.append({"n_crane": n_crane, "n_ship": n_ship,
                            "n_total": len(last_tracks)})

    elapsed = time.time() - t0
    cap.release(); writer.close()

    final_name = os.path.basename(actual_path)
    shutil.copy(actual_path, final_name)
    print(f"\n  [OK] Video -> {final_name}")
    print(f"       {total} frames in {elapsed:.1f}s  ({total/max(elapsed, 1e-6):.1f} fps)")

    dash_path = os.path.join(CFG["output_dir"], CFG["output_dashboard"])
    generate_dashboard(history, heatmap.data, dash_path)
    shutil.copy(dash_path, CFG["output_dashboard"])

    print("\n" + "=" * 70)
    print("  DONE  --  PORT BULK CARRIER INTELLIGENCE v6.0")
    print(f"  Video     -> {final_name}")
    print(f"  Dashboard -> {CFG['output_dashboard']}")
    if final_name.endswith(".avi"):
        print("  NOTE: .avi output — open with VLC or Windows Media Player")
    print(f"  Speed     -> {total/max(elapsed, 1e-6):.1f} fps  ({elapsed:.1f}s)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run()