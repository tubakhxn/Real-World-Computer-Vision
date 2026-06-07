#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║   VISION-ONLY AUTONOMOUS DRIVING SIMULATOR  ── FINAL v4             ║
║   Dev/Creator: tubakhxn                                             ║
║   Clean detection overlay + Full-featured Tesla-style HUD           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import subprocess, sys, os, warnings, glob, math, time
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
INPUT_VIDEO  = ""
OUTPUT_VIDEO = "autonomous_drive_OUTPUT.mp4"
DASHBOARD    = "autonomous_drive_DASHBOARD.png"
HUD_W        = 480       # wider HUD panel for more content
# ═══════════════════════════════════════════════════════════════════════

REQUIRED = ["opencv-python","numpy","matplotlib","ultralytics",
            "torch","torchvision","Pillow","tqdm","scipy"]

def auto_install(pkgs):
    for p in pkgs:
        try: __import__(p.replace("-","_").split("[")[0])
        except ImportError:
            print(f"  [INSTALL] {p} …")
            subprocess.check_call([sys.executable,"-m","pip","install",p,"-q"],
                                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

print("━"*62)
print("  VISION-ONLY AUTONOMOUS DRIVING  ── v4  |  tubakhxn")
print("━"*62)
print("[INIT] Checking deps …")
auto_install(REQUIRED)

import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

# ── auto-detect input video ────────────────────────────────────────────
def find_video(hint: str) -> str:
    if hint and os.path.exists(hint):
        return hint
    for pattern in ["driving_input.*","dashcam.*","road.*","drive.*",
                    "input.*","video.*","*.mp4","*.avi","*.mov","*.mkv"]:
        for c in glob.glob(pattern):
            if "OUTPUT" not in c.upper() and os.path.isfile(c):
                return c
    return ""

# ── YOLO class config ──────────────────────────────────────────────────
DET_CLASSES = {
    0:  {"name":"Pedestrian",  "color":(0,255,100),   "real_h":1.7, "risk":"high"},
    1:  {"name":"Bicycle",     "color":(255,200,0),   "real_h":1.0, "risk":"medium"},
    2:  {"name":"Car",         "color":(0,180,255),   "real_h":1.5, "risk":"medium"},
    3:  {"name":"Motorcycle",  "color":(255,100,0),   "real_h":1.2, "risk":"high"},
    5:  {"name":"Bus",         "color":(0,50,255),    "real_h":3.5, "risk":"low"},
    7:  {"name":"Truck",       "color":(180,100,255), "real_h":3.2, "risk":"low"},
    9:  {"name":"TrafficLight","color":(255,255,0),   "real_h":0.5, "risk":"info"},
    11: {"name":"StopSign",    "color":(0,0,255),     "real_h":0.7, "risk":"info"},
}
FOCAL = 900.0

@dataclass
class Det:
    cls: int; label: str; bbox: Tuple; conf: float
    dist: float = 0.0; risk: str = "low"; color: Tuple = (0,255,0)

@dataclass
class Ego:
    speed: float = 0.0
    steer: float = 0.0
    throttle: float = 0.5
    brake: float = 0.0
    accel_g: float = 0.0        # longitudinal g-force
    lateral_g: float = 0.0     # lateral g-force

@dataclass
class Decision:
    action: str = "CRUISE"
    throttle: float = 0.5
    brake: float = 0.0
    steer: float = 0.0
    target_spd: float = 50.0
    reason: str = ""
    risk_lvl: str = "LOW"
    color: Tuple = (0,200,50)


# ══════════════════════════════════════════════════════════════════════
#  PERCEPTION  (detection only — no lane drawing)
# ══════════════════════════════════════════════════════════════════════

class Perception:
    def __init__(self):
        self._lane_off_hist = deque([0.0]*10, maxlen=10)

    def distance(self, bbox, cls):
        h = bbox[3] - bbox[1]
        if h < 4: return 999.0
        rh = DET_CLASSES.get(cls, {"real_h":1.5})["real_h"]
        return round((rh * FOCAL) / h, 1)

    def lane_offset_estimate(self, frame) -> float:
        """Lightweight lane offset via edge histogram — no drawing."""
        H, W = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (7,7), 0)
        edges = cv2.Canny(blur, 40, 120)
        roi_y = int(H * 0.60)
        strip = edges[roi_y:, :]
        col_sum = strip.sum(axis=0).astype(float)
        cx = W // 2
        left_mass  = col_sum[:cx].sum()
        right_mass = col_sum[cx:].sum()
        total = left_mass + right_mass + 1e-6
        offset = (right_mass - left_mass) / total * 0.6
        self._lane_off_hist.append(offset)
        return float(np.mean(self._lane_off_hist))

    def collision_risk(self, dets, W, H) -> float:
        risk = 0.0
        pw = W // 5
        for d in dets:
            cx_ = (d.bbox[0] + d.bbox[2]) // 2
            cy_ = (d.bbox[1] + d.bbox[3]) // 2
            in_path = (W//2 - pw < cx_ < W//2 + pw) and cy_ > H * .45
            dr  = max(0, 1 - d.dist / 55.0)
            or_ = {"high":1.0,"medium":0.65,"low":0.3,"info":0.0}.get(d.risk, 0)
            risk = max(risk, dr * or_ * (1.6 if in_path else 0.6))
        return min(1.0, risk)


# ══════════════════════════════════════════════════════════════════════
class DecisionEngine:
    def __init__(self):
        self.speed = 35.0
        self.steer_buf = deque([0.0]*8, maxlen=8)
        self.prev_speed = 35.0
        self.spd_log:  List[float] = []
        self.risk_log: List[float] = []
        self.lane_log: List[float] = []
        self.dec_log:  List[str]   = []
        self.det_log:  List[Dict]  = []

    def decide(self, dets, lane_off, cr, lane_conf) -> Decision:
        d = Decision()
        nearest = min((x.dist for x in dets if x.risk in ("high","medium")),
                      default=999.0)

        if nearest < 7.0 or cr > 0.88:
            d.action="EMERGENCY BRAKE"; d.brake=1.0; d.throttle=0
            d.target_spd=0; d.risk_lvl="CRITICAL"; d.color=(0,0,255)
            d.reason=f"Object at {nearest:.1f}m"
        elif nearest < 18.0 or cr > 0.55:
            d.action="BRAKING"; d.brake=0.5+(18-nearest)/30
            d.throttle=0; d.target_spd=max(8,nearest*1.4)
            d.risk_lvl="HIGH"; d.color=(0,80,255)
            d.reason=f"Hazard {nearest:.1f}m ahead"
        elif nearest < 32.0 or cr > 0.28:
            d.action="SLOW DOWN"; d.throttle=0.25
            d.target_spd=28; d.risk_lvl="MEDIUM"
            d.color=(0,165,255); d.reason="Traffic ahead"
        elif abs(lane_off) > 0.18:
            d.action="LANE CORRECT"; d.throttle=0.55
            d.target_spd=48; d.risk_lvl="LOW"
            d.color=(255,200,0); d.reason=f"Drift {lane_off:+.2f}"
        elif self.speed < 47:
            d.action="ACCELERATE"; d.throttle=0.82
            d.target_spd=52; d.risk_lvl="LOW"
            d.color=(50,220,50); d.reason="Speeding up to 50"
        else:
            d.action="CRUISE"; d.throttle=0.5
            d.target_spd=50; d.risk_lvl="LOW"
            d.color=(100,255,100); d.reason="Maintaining speed"

        self.steer_buf.append(-lane_off * 0.38)
        d.steer = float(np.clip(np.mean(self.steer_buf), -1, 1))

        if d.brake > 0: self.speed = max(0, self.speed - d.brake * 4)
        else:           self.speed = min(d.target_spd, self.speed + d.throttle * 2)

        return d


# ══════════════════════════════════════════════════════════════════════
#  HUD  — full-featured, uses every pixel of the 480px panel
# ══════════════════════════════════════════════════════════════════════

class HUD:
    def __init__(self, W, H, hw=HUD_W):
        self.W = W; self.H = H; self.hw = hw
        # color palette
        self.DARK   = (8, 10, 16)
        self.PANEL  = (14, 17, 26)
        self.PANEL2 = (18, 22, 34)
        self.ACC    = (0, 220, 255)       # cyan accent
        self.GREEN  = (40, 220, 70)
        self.AMBER  = (0, 160, 255)
        self.RED    = (0, 40, 255)
        self.WHITE  = (235, 238, 245)
        self.GREY   = (90, 100, 118)
        self.YELLOW = (0, 210, 255)

    @staticmethod
    def txt(img, text, x, y, scale=0.55, color=(220,220,225),
            bold=False, shadow=True):
        th = 2 if bold else 1
        if shadow:
            cv2.putText(img, text, (x+1, y+1),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), th+1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)

    @staticmethod
    def bar_h(img, x, y, w, h, val, maxv, color, bg=(18,22,34), radius=3):
        cv2.rectangle(img, (x,y), (x+w, y+h), bg, -1)
        filled = int(w * min(val, maxv) / max(maxv, 1e-6))
        if filled > 0:
            cv2.rectangle(img, (x,y), (x+filled, y+h), color, -1)

    @staticmethod
    def bar_v(img, x, y, w, h, val, maxv, color, bg=(18,22,34)):
        """Vertical bar — fills from bottom up."""
        cv2.rectangle(img, (x,y), (x+w, y+h), bg, -1)
        filled = int(h * min(val, maxv) / max(maxv, 1e-6))
        if filled > 0:
            cv2.rectangle(img, (x, y+h-filled), (x+w, y+h), color, -1)

    # ── section divider ────────────────────────────────────────────────
    def divider(self, p, y, label=""):
        cv2.line(p, (6,y), (self.hw-6,y), (28,34,50), 1)
        if label:
            (tw,_),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(p, (self.hw//2-tw//2-6,y-8),(self.hw//2+tw//2+6,y+8),(14,17,26),-1)
            self.txt(p, label, self.hw//2-tw//2, y+5, 0.40, self.GREY, shadow=False)

    # ── arc speedometer ────────────────────────────────────────────────
    def speedo(self, p, cx, cy, r, spd, maxs=120):
        # outer ring
        cv2.ellipse(p,(cx,cy),(r+4,r+4),0,0,360,(22,28,42),1)
        # track arc
        cv2.ellipse(p,(cx,cy),(r,r),0,135,405,(30,35,50),r-2)
        # value arc
        a_end = int(135 + (min(spd,maxs)/maxs) * 270)
        if spd < 40: arc_c = self.GREEN
        elif spd < 80: arc_c = self.AMBER
        else: arc_c = self.RED
        if a_end > 135:
            cv2.ellipse(p,(cx,cy),(r,r),0,135,a_end,arc_c,r-2)
        # tick marks
        for i in range(0,13):
            a = math.radians(135 + i*22.5 - 90)
            r1 = r+2; r2 = r-5 if i%4==0 else r-3
            x1,y1 = int(cx+r1*math.cos(a)), int(cy+r1*math.sin(a))
            x2,y2 = int(cx+r2*math.cos(a)), int(cy+r2*math.sin(a))
            cv2.line(p,(x1,y1),(x2,y2),(60,70,90),1,cv2.LINE_AA)
        # needle
        na = math.radians(135 + (min(spd,maxs)/maxs)*270 - 90)
        nx = int(cx + (r-8)*math.cos(na))
        ny = int(cy + (r-8)*math.sin(na))
        cv2.line(p,(cx,cy),(nx,ny),(255,255,255),2,cv2.LINE_AA)
        cv2.circle(p,(cx,cy),5,(160,170,190),-1)
        # number
        spd_str = f"{spd:.0f}"
        (tw,th_),_ = cv2.getTextSize(spd_str, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(p,spd_str,(cx-tw//2,cy+th_//2+2),
                    cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(p,"km/h",(cx-16,cy+th_//2+18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.34,(110,120,140),1,cv2.LINE_AA)
        cv2.putText(p,"SPEED",(cx-22,cy-r-14),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,self.ACC,1,cv2.LINE_AA)

    # ── steering wheel ─────────────────────────────────────────────────
    def wheel(self, p, cx, cy, r, steer):
        cv2.circle(p,(cx,cy),r,(40,48,66),2)
        cv2.circle(p,(cx,cy),r-4,(16,18,28),-1)
        for sa in [0,120,240]:
            a = math.radians(steer*195+sa-90)
            x1,y1 = int(cx+(r-2)*math.cos(a)),int(cy+(r-2)*math.sin(a))
            cv2.line(p,(x1,y1),(2*cx-x1,2*cy-y1),(70,78,100),3)
        cv2.circle(p,(cx,cy),7,(48,56,76),-1)
        na = math.radians(steer*90-90)
        nx,ny = int(cx+(r-3)*math.cos(na)),int(cy+(r-3)*math.sin(na))
        cv2.line(p,(cx,cy),(nx,ny),(240,200,40),2,cv2.LINE_AA)
        steer_deg = int(steer*540)
        (tw,_),_ = cv2.getTextSize(f"{steer_deg:+d}°",cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
        self.txt(p,f"{steer_deg:+d}°",cx-tw//2,cy+r+16,0.38,(160,165,185),shadow=False)
        cv2.putText(p,"STEER",(cx-22,cy-r-14),
                    cv2.FONT_HERSHEY_SIMPLEX,0.38,self.ACC,1,cv2.LINE_AA)

    # ── 3D G-force ball ────────────────────────────────────────────────
    def gforce_ball(self, p, cx, cy, r, long_g, lat_g):
        """G-force indicator: ball moves inside circle based on acceleration."""
        # background
        cv2.circle(p,(cx,cy),r,(20,24,36),-1)
        cv2.circle(p,(cx,cy),r,(40,50,70),1)
        # rings
        for ri in [r//3, 2*r//3]:
            cv2.circle(p,(cx,cy),ri,(28,34,50),1)
        # crosshairs
        cv2.line(p,(cx-r,cy),(cx+r,cy),(30,38,55),1)
        cv2.line(p,(cx,cy-r),(cx,cy+r),(30,38,55),1)
        # clamp g values and compute ball pos
        lg = np.clip(long_g, -1.0, 1.0)
        la = np.clip(lat_g,  -1.0, 1.0)
        bx = int(cx + la * (r-8))
        by = int(cy - lg * (r-8))   # negative: forward accel goes up
        # ball with glow
        glow_c = (0,100,255) if lg < -0.1 else (0,200,80) if lg > 0.1 else (0,180,220)
        for radius_g, alpha in [(10,60),(8,120),(6,200)]:
            overlay = p.copy()
            cv2.circle(overlay,(bx,by),radius_g,glow_c,-1)
            cv2.addWeighted(p,1-alpha/255,overlay,alpha/255,0,p)
        cv2.circle(p,(bx,by),5,glow_c,-1)
        cv2.circle(p,(bx,by),5,(255,255,255),1)
        # labels
        self.txt(p,"G-FORCE",cx-24,cy-r-14,0.38,self.ACC,shadow=False)
        self.txt(p,f"L:{long_g:+.2f}g",cx-r-2,cy+r+16,0.34,(160,165,185),shadow=False)
        self.txt(p,f"T:{lat_g:+.2f}g",cx+18,cy+r+16,0.34,(160,165,185),shadow=False)

    # ── 3D pedal visualizer ────────────────────────────────────────────
    def pedals_3d(self, p, x, y, w, h, throttle, brake):
        """3D-ish pedal blocks showing throttle and brake pressure."""
        pad = 8
        pw  = (w - pad*3) // 2   # pedal width

        def draw_pedal(px, py, pw_, ph, val, label, col_on, col_off=(22,28,42)):
            filled_h = int(ph * val)
            # pedal body (3D effect)
            # shadow
            pts = np.array([[px+4,py+4],[px+pw_+4,py+4],[px+pw_+4,py+ph+4],[px+4,py+ph+4]])
            cv2.fillPoly(p,[pts],(8,10,16))
            # back face
            cv2.rectangle(p,(px,py),(px+pw_,py+ph),col_off,-1)
            # side face (3D right)
            side = np.array([[px+pw_,py],[px+pw_+4,py+4],[px+pw_+4,py+ph+4],[px+pw_,py+ph]])
            lighter = tuple(min(255,int(c*0.6)) for c in col_off)
            cv2.fillPoly(p,[side],lighter)
            # top face (3D)
            top = np.array([[px,py],[px+pw_,py],[px+pw_+4,py+4],[px+4,py+4]])
            darker = tuple(min(255,int(c*0.8)) for c in col_off)
            cv2.fillPoly(p,[top],darker)
            # filled bar (active)
            if filled_h > 0:
                fy = py + ph - filled_h
                cv2.rectangle(p,(px,fy),(px+pw_,py+ph),col_on,-1)
                # glow overlay on active
                glow = p[fy:py+ph, px:px+pw_].copy()
                white_overlay = np.full_like(glow, col_on)
                cv2.addWeighted(glow,0.6,white_overlay,0.4,0,p[fy:py+ph, px:px+pw_])
            # border
            cv2.rectangle(p,(px,py),(px+pw_,py+ph),(50,58,78),1)
            # value text
            pct = f"{int(val*100)}%"
            (tw,_),_ = cv2.getTextSize(pct, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            self.txt(p,pct,px+pw_//2-tw//2,py+ph+16,0.45,
                     col_on if val>0.05 else self.GREY,shadow=False)
            # label
            (tw2,_),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            self.txt(p,label,px+pw_//2-tw2//2,py-6,0.38,self.GREY,shadow=False)

        draw_pedal(x+pad,       y, pw, h, throttle, "THROTTLE",
                   (0,200,80),  (18,28,22))
        draw_pedal(x+pad*2+pw,  y, pw, h, brake,    "BRAKE",
                   (0,50,220),  (22,18,28))

    # ── bird's eye view ───────────────────────────────────────────────
    def bev(self, p, ox, oy, sz, dets, ego, dec):
        bev = np.full((sz, sz, 3), (10,12,18), np.uint8)
        c   = sz // 2

        # road surface
        cv2.rectangle(bev,(c-32,0),(c+32,sz),(20,22,32),-1)
        # dashed center line
        dash_len = 14
        for dy in range(0, sz, dash_len*2):
            cv2.line(bev,(c,dy),(c,min(sz-1,dy+dash_len)),(50,50,30),1)
        # lane markers
        cv2.line(bev,(c-32,0),(c-32,sz),(35,40,55),1)
        cv2.line(bev,(c+32,0),(c+32,sz),(35,40,55),1)

        # ego car (3D box style)
        car_w,car_h = 14,22
        ex,ey = c, sz-36
        # car shadow
        cv2.rectangle(bev,(ex-car_w//2+2,ey-car_h+2),(ex+car_w//2+2,ey+2),(5,6,10),-1)
        # car body
        cv2.rectangle(bev,(ex-car_w//2,ey-car_h),(ex+car_w//2,ey),(0,180,70),-1)
        # windscreen
        cv2.rectangle(bev,(ex-car_w//2+2,ey-car_h+3),(ex+car_w//2-2,ey-car_h+9),(0,240,90),-1)
        # heading arrow
        arr_len = int(20 + ego.speed * 0.3)
        ax_end  = int(ex + math.sin(dec.steer*0.8)*arr_len*0.4)
        ay_end  = ey - car_h - arr_len
        cv2.arrowedLine(bev,(ex,ey-car_h),(ax_end,max(2,ay_end)),
                        (0,220,100),2,cv2.LINE_AA,tipLength=0.2)

        # detected objects
        used_x = {}
        for di, d in enumerate(sorted(dets, key=lambda x: x.dist)):
            dist_frac = int((1.0 - min(d.dist, 80)/80.0) * (sz-50)) + 10
            # spread horizontally by detection index
            spread = (di - len(dets)//2) * 18
            bx = np.clip(c + spread, 8, sz-8)
            col = d.color
            obj_w,obj_h = 10,7
            # shadow
            cv2.rectangle(bev,(bx-obj_w//2+1,dist_frac-obj_h+1),
                          (bx+obj_w//2+1,dist_frac+1),(5,6,10),-1)
            cv2.rectangle(bev,(bx-obj_w//2,dist_frac-obj_h),
                          (bx+obj_w//2,dist_frac),col,-1)
            # threat line from ego to object
            threat = {"high":True,"medium": d.dist < 25}.get(d.risk, False)
            if threat:
                cv2.line(bev,(ex,ey-car_h),(bx,dist_frac),(0,0,180),1,cv2.LINE_AA)

        # distance rings
        for dist_m in [15,30,50]:
            ry = int((1.0 - dist_m/80.0) * (sz-50)) + 10
            cv2.line(bev,(c-36,ry),(c+36,ry),(30,35,48),1)
            self.txt(bev,f"{dist_m}m",c+38,ry+5,0.28,(50,60,80),shadow=False)

        cv2.rectangle(bev,(0,0),(sz-1,sz-1),(35,42,60),1)
        cv2.putText(bev,"BIRD'S-EYE",(4,12),
                    cv2.FONT_HERSHEY_SIMPLEX,0.30,(60,75,100),1,cv2.LINE_AA)
        p[oy:oy+sz, ox:ox+sz] = bev

    # ── threat meter (vertical) ───────────────────────────────────────
    def threat_meter(self, p, x, y, w, h, cr):
        """Vertical threat level bar with zones."""
        cv2.rectangle(p,(x,y),(x+w,y+h),(14,16,24),-1)
        segments = [
            (0.0, 0.28, (0,180,60)),
            (0.28,0.55, (0,140,220)),
            (0.55,0.88, (0,80,255)),
            (0.88,1.00, (0,20,220)),
        ]
        for lo,hi,col in segments:
            sy = y + int((1-hi)*h)
            ey_ = y + int((1-lo)*h)
            alpha_c = tuple(min(255,int(c*0.35)) for c in col)
            cv2.rectangle(p,(x,sy),(x+w,ey_),alpha_c,-1)
        # filled level
        filled_h = int(h * cr)
        if filled_h > 0:
            fy = y + h - filled_h
            # gradient-ish by zone
            cr_col = (0,180,60) if cr<0.28 else (0,120,240) if cr<0.55 else (0,50,230) if cr<0.88 else (0,10,220)
            cv2.rectangle(p,(x+2,fy),(x+w-2,y+h),cr_col,-1)
        # border
        cv2.rectangle(p,(x,y),(x+w,y+h),(50,60,80),1)
        # label
        lbl = "SAFE" if cr<0.28 else "CAUTION" if cr<0.55 else "HIGH" if cr<0.88 else "CRITICAL"
        (tw,_),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
        cr_col2 = (0,200,80) if cr<0.28 else (0,160,255) if cr<0.55 else (0,80,255) if cr<0.88 else (0,30,255)
        self.txt(p, lbl, x+w//2-tw//2, y+h+14, 0.32, cr_col2, shadow=False)

    # ── full render ───────────────────────────────────────────────────
    def render(self, frame, dets, lane_off, lane_conf, dec, ego, cr, fidx, total):
        H, W = frame.shape[:2]
        hw   = self.hw
        out  = np.zeros((H, W+hw, 3), np.uint8)
        out[:, :W] = frame

        # ── detection boxes ONLY on video feed ───────────────────────
        for d in dets:
            x1,y1,x2,y2 = d.bbox
            c = d.color
            thick = 3 if d.risk == "high" else 2

            # main box
            cv2.rectangle(out, (x1,y1), (x2,y2), c, thick, cv2.LINE_AA)

            # corner tick marks
            tl = min(18, (x2-x1)//4, (y2-y1)//4)
            for sx,sy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(out,(sx,sy),(sx+dx*tl,sy),c,2,cv2.LINE_AA)
                cv2.line(out,(sx,sy),(sx,sy+dy*tl),c,2,cv2.LINE_AA)

            # distance warning dot for nearby threats
            if d.dist < 15 and d.risk in ("high","medium"):
                pulse_r = max(4, int(10 - d.dist*0.5))
                cv2.circle(out,(x1+6,y1+6),pulse_r,self.RED,-1,cv2.LINE_AA)

            # label pill
            lbl = f"{d.label}  {d.dist:.0f}m  {d.conf:.0%}"
            (tw,th_),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(out,(x1,y1-th_-10),(x1+tw+8,y1),(0,0,0),-1)
            cv2.rectangle(out,(x1,y1-th_-10),(x1+tw+8,y1),c,1)
            cv2.putText(out,lbl,(x1+4,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.50,c,1,cv2.LINE_AA)

        # ═══════════════════════════════════════════════════════════
        #  HUD PANEL
        # ═══════════════════════════════════════════════════════════
        panel = out[:, W:]
        panel[:] = self.DARK
        # left border accent
        cv2.rectangle(out,(W,0),(W+3,H),self.ACC,-1)

        T = self.txt

        # ─── HEADER ───────────────────────────────────────────────
        cv2.rectangle(panel,(0,0),(hw,60),(12,15,24),-1)
        cv2.rectangle(panel,(0,60),(hw,62),self.ACC,-1)
        T(panel,"AUTONOMY HUD",    8, 28,  0.78, self.ACC,  bold=True,  shadow=False)
        T(panel,"tubakhxn | v4",   8, 50,  0.36, self.GREY, shadow=False)
        T(panel,f"Frame {fidx}/{total}", hw-110,50, 0.36,self.GREY,shadow=False)

        y = 68

        # ─── DIALS ROW: speed + steering ──────────────────────────
        dial_r   = 56
        spd_cx   = hw // 4
        str_cx   = (3*hw) // 4
        dial_cy  = y + dial_r + 14
        self.speedo(panel, spd_cx,  dial_cy, dial_r, ego.speed)
        self.wheel (panel, str_cx,  dial_cy, dial_r, dec.steer)

        y = dial_cy + dial_r + 28
        cv2.line(panel,(4,y),(hw-4,y),(25,30,46),1)
        y += 10

        # ─── AI DECISION BOX ──────────────────────────────────────
        box_h = 56
        cv2.rectangle(panel,(6,y),(hw-6,y+box_h),(14,16,26),-1)
        # left color bar
        cv2.rectangle(panel,(6,y),(10,y+box_h),dec.color,-1)
        cv2.rectangle(panel,(6,y),(hw-6,y+box_h),dec.color,2)
        T(panel, dec.action,          16, y+28, 0.78, dec.color, bold=True,  shadow=False)
        T(panel, dec.reason[:38],     16, y+48, 0.40, self.WHITE, shadow=False)
        # risk badge
        rw = 90
        rc = {"LOW":(0,180,70),"MEDIUM":(0,140,220),
              "HIGH":(0,70,240),"CRITICAL":(0,20,220)}.get(dec.risk_lvl,(100,100,100))
        cv2.rectangle(panel,(hw-rw-8,y+8),(hw-8,y+30),rc,-1)
        (tw,_),_=cv2.getTextSize(dec.risk_lvl,cv2.FONT_HERSHEY_SIMPLEX,0.40,1)
        T(panel,dec.risk_lvl,hw-rw-8+(rw-tw)//2,y+24,0.40,(0,0,0),bold=True,shadow=False)
        y += box_h + 12

        # ─── G-FORCE + PEDALS ROW ────────────────────────────────
        gf_sz  = 90
        ped_x  = gf_sz + 24
        ped_w  = hw - ped_x - 8
        ped_h  = gf_sz - 20

        T(panel,"G-FORCE DYNAMICS",6,y+14,0.44,self.ACC,bold=True,shadow=False)
        y += 18

        # g-force ball
        self.gforce_ball(panel, gf_sz//2+4, y+gf_sz//2+4, gf_sz//2-4,
                         ego.accel_g, ego.lateral_g)

        # 3D pedals to the right of g-ball
        self.pedals_3d(panel, ped_x, y+18, ped_w, ped_h,
                       dec.throttle, dec.brake)

        y += gf_sz + 28
        cv2.line(panel,(4,y),(hw-4,y),(25,30,46),1)
        y += 10

        # ─── LANE CONFIDENCE ──────────────────────────────────────
        T(panel,"LANE CONFIDENCE",6,y+14,0.44,self.ACC,bold=True,shadow=False)
        lc_col = self.GREEN if lane_conf>.6 else (self.AMBER if lane_conf>.3 else self.RED)
        self.bar_h(panel,6,y+18,hw-12,16,lane_conf,1.0,lc_col)
        pct_str = f"{lane_conf*100:.0f}%"
        (tw,_),_=cv2.getTextSize(pct_str,cv2.FONT_HERSHEY_SIMPLEX,0.40,1)
        T(panel,pct_str,hw//2-tw//2,y+30,0.40,self.WHITE,shadow=False)
        # offset bar
        T(panel,f"Offset: {lane_off:+.3f}",6,y+48,0.40,
          self.AMBER if abs(lane_off)>.18 else self.WHITE,shadow=False)
        y += 58
        cv2.line(panel,(4,y),(hw-4,y),(25,30,46),1)
        y += 10

        # ─── DETECTIONS PANEL ────────────────────────────────────
        T(panel,"DETECTIONS",6,y+14,0.44,self.ACC,bold=True,shadow=False)
        y += 20

        peds   = [d for d in dets if d.label=="Pedestrian"]
        bikes  = [d for d in dets if d.label in ("Bicycle","Motorcycle")]
        cars   = [d for d in dets if d.label in ("Car","Truck","Bus")]
        lights = [d for d in dets if d.label in ("TrafficLight","StopSign")]

        def det_row(label, count, closest, col, row_y):
            # icon square
            sq_col = col if count>0 else (30,35,50)
            cv2.rectangle(panel,(6,row_y),(20,row_y+16),sq_col,-1)
            T(panel,str(count),8,row_y+13,0.38,self.WHITE,shadow=False)
            T(panel,label,24,row_y+13,0.44,col if count>0 else self.GREY,shadow=False)
            if count>0 and closest<999:
                dist_s = f"{closest:.0f}m"
                (tw,_),_=cv2.getTextSize(dist_s,cv2.FONT_HERSHEY_SIMPLEX,0.38,1)
                T(panel,dist_s,hw-tw-8,row_y+13,0.38,
                  self.RED if closest<15 else self.AMBER if closest<30 else self.WHITE,
                  shadow=False)

        closest_ped   = min((d.dist for d in peds),   default=999)
        closest_bike  = min((d.dist for d in bikes),  default=999)
        closest_car   = min((d.dist for d in cars),   default=999)

        det_row("Pedestrians",len(peds),  closest_ped,  self.GREEN,  y)
        y += 20
        det_row("Cyclists",   len(bikes), closest_bike,(255,180,0),  y)
        y += 20
        det_row("Vehicles",   len(cars),  closest_car,  self.ACC,    y)
        y += 20
        det_row("Signals",    len(lights),999,          (255,230,0),  y)
        y += 20
        # total
        cv2.line(panel,(20,y),(hw-20,y),(30,36,52),1)
        y += 6
        T(panel,f"Total detected: {len(dets)}",6,y+14,0.48,self.WHITE,bold=True,shadow=False)

        nearest_all = min((d.dist for d in dets), default=999.0)
        nearest_s = f"Nearest: {nearest_all:.1f}m"
        (tw,_),_=cv2.getTextSize(nearest_s,cv2.FONT_HERSHEY_SIMPLEX,0.44,1)
        T(panel,nearest_s,hw-tw-6,y+14,0.44,
          self.RED if nearest_all<15 else self.AMBER if nearest_all<30 else self.WHITE,
          shadow=False)
        y += 22
        cv2.line(panel,(4,y),(hw-4,y),(25,30,46),1)
        y += 10

        # ─── THREAT + BIRD'S-EYE side by side ─────────────────────
        remaining = H - y - 8
        if remaining > 80:
            # threat meter on left (narrow)
            tm_w = 24
            tm_h = remaining - 30
            T(panel,"THREAT",6,y+12,0.34,self.ACC,shadow=False)
            self.threat_meter(panel, 6, y+18, tm_w, tm_h, cr)

            # bird's eye on right
            bev_sz = min(remaining-10, hw - tm_w - 28)
            bev_x  = tm_w + 18
            T(panel,"BIRD'S-EYE",bev_x,y+12,0.36,self.ACC,shadow=False)
            self.bev(panel, bev_x, y+18, bev_sz, dets, ego, dec)

        # ─── BOTTOM PROGRESS BAR ─────────────────────────────────
        prog = fidx / max(1, total)
        cv2.rectangle(out,(0,H-20),(W+hw,H),(6,8,12),-1)
        cv2.rectangle(out,(0,H-4),(int((W+hw)*prog),H),self.ACC,-1)
        cv2.putText(out,
            f"  {ego.speed:.0f} km/h  |  {dec.action}  |  Risk: {dec.risk_lvl}  |  "
            f"Objects: {len(dets)}  |  tubakhxn",
            (6,H-6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70,80,100), 1, cv2.LINE_AA)

        return out


# ══════════════════════════════════════════════════════════════════════
#  ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════

def make_dashboard(spd_log, risk_log, lane_log, dec_log, det_log, path):
    print("[DASHBOARD] Generating …")
    from collections import Counter
    fig = plt.figure(figsize=(22,12), facecolor="#05080f")
    fig.suptitle("VISION-ONLY AUTONOMOUS DRIVING  ──  ANALYTICS DASHBOARD",
                 color="#00e5ff", fontsize=17, fontweight="bold", y=.99,
                 fontfamily="monospace")
    fig.text(.99,.99,"tubakhxn | v4",color="#2a3050",fontsize=9,ha="right",va="top",
             fontfamily="monospace")
    gs = fig.add_gridspec(3,5,hspace=.50,wspace=.40,
                          left=.05,right=.98,top=.94,bottom=.06)
    C = {"bg":"#0d1117","p":"#10141c","a":"#00e5ff","g":"#00e676",
         "r":"#ff1744","o":"#ff9100","t":"#cfd8dc","gr":"#1e2430",
         "y":"#ffea00","m":"#e040fb"}

    def sax(ax, title=""):
        ax.set_facecolor(C["p"])
        for sp in ax.spines.values(): sp.set_color(C["gr"])
        ax.tick_params(colors=C["t"], labelsize=8)
        if title:
            ax.set_title(title,color=C["a"],fontsize=10,
                         fontfamily="monospace",pad=6)

    F = list(range(len(spd_log)))

    # Speed
    ax1 = fig.add_subplot(gs[0,:2]); sax(ax1,"Speed Over Time (km/h)")
    if F:
        ax1.fill_between(F,spd_log,alpha=.20,color=C["g"])
        ax1.plot(F,spd_log,color=C["g"],lw=1.5)
        ax1.axhline(50,color=C["a"],ls="--",lw=.8,alpha=.5,label="Target 50 km/h")
    ax1.legend(facecolor=C["p"],edgecolor=C["gr"],labelcolor=C["t"],fontsize=8)
    ax1.set_xlabel("Frame",color=C["t"],fontsize=8)
    ax1.yaxis.grid(True,color=C["gr"],alpha=.4)

    # Collision risk
    ax2 = fig.add_subplot(gs[0,2:4]); sax(ax2,"Collision Risk Score")
    if F:
        ax2.fill_between(F,risk_log,alpha=.20,color=C["r"])
        ax2.plot(F,risk_log,color=C["r"],lw=1.5)
        ax2.axhline(.55,color=C["o"],ls="--",lw=.8,alpha=.6,label="Brake zone")
        ax2.axhline(.88,color=C["r"],ls="--",lw=.8,alpha=.8,label="E-stop zone")
        ax2.set_ylim(0,1.1)
    ax2.legend(facecolor=C["p"],edgecolor=C["gr"],labelcolor=C["t"],fontsize=8)
    ax2.yaxis.grid(True,color=C["gr"],alpha=.4)

    # KPI box
    ax_kpi = fig.add_subplot(gs[0,4])
    ax_kpi.set_facecolor(C["p"])
    for sp in ax_kpi.spines.values(): sp.set_visible(False)
    ax_kpi.set(xlim=(0,1),ylim=(0,1),xticks=[],yticks=[])
    ax_kpi.set_title("KPIs",color=C["a"],fontsize=10,fontfamily="monospace",pad=6)
    kpis=[
        ("Avg Speed",    f"{np.mean(spd_log):.1f} km/h",  C["g"]),
        ("Peak Risk",    f"{max(risk_log):.2f}",           C["r"]),
        ("Lane Conf",    f"{np.mean(lane_log)*100:.0f}%",  C["a"]),
        ("Total Objects",str(sum(d.get("total",0) for d in det_log)), C["o"]),
    ]
    for i,(lb,v,c_) in enumerate(kpis):
        y_ = .82 - i*.22
        ax_kpi.add_patch(FancyBboxPatch((.04,y_-.10),.92,.20,
                         boxstyle="round,pad=.02",facecolor=C["bg"],edgecolor=c_,lw=1.5))
        ax_kpi.text(.5,y_+.03,v,color=c_,fontsize=13,ha="center",va="center",
                   fontfamily="monospace",fontweight="bold")
        ax_kpi.text(.5,y_-.04,lb,color=C["t"],fontsize=7,ha="center",va="center")

    # Decision distribution
    ax3 = fig.add_subplot(gs[1,0]); sax(ax3,"AI Decision Distribution")
    if dec_log:
        from collections import Counter
        cnt=Counter(dec_log); lbs=list(cnt.keys()); vs=list(cnt.values())
        cols=[C["g"] if "CRUISE" in l else C["o"] if "ACCEL" in l
              else C["r"] if "BRAKE" in l else C["m"] if "CORRECT" in l
              else C["a"] for l in lbs]
        ax3.pie(vs,labels=lbs,colors=cols,autopct="%1.0f%%",startangle=90,
                textprops={"color":C["t"],"fontsize":7},
                wedgeprops={"edgecolor":C["bg"],"linewidth":2})

    # Lane confidence
    ax4 = fig.add_subplot(gs[1,1]); sax(ax4,"Lane Confidence Over Time")
    if lane_log:
        ax4.fill_between(F,lane_log,alpha=.25,color=C["a"])
        ax4.plot(F,lane_log,color=C["a"],lw=1.2)
        ax4.set_ylim(0,1.1)
        ax4.axhline(.6,color=C["g"],ls="--",lw=.8,alpha=.5,label="Good")
    ax4.legend(facecolor=C["p"],edgecolor=C["gr"],labelcolor=C["t"],fontsize=8)
    ax4.yaxis.grid(True,color=C["gr"],alpha=.4)

    # Detections stacked
    ax5 = fig.add_subplot(gs[1,2:4]); sax(ax5,"Detections Per Frame")
    if det_log:
        pl=[d.get("peds",0) for d in det_log]
        cl=[d.get("cars",0) for d in det_log]
        ax5.stackplot(F,pl,cl,colors=[C["g"],C["a"]],alpha=.75,
                      labels=["Pedestrians","Vehicles"])
    ax5.legend(facecolor=C["p"],edgecolor=C["gr"],labelcolor=C["t"],fontsize=8)
    ax5.yaxis.grid(True,color=C["gr"],alpha=.4)
    ax5.set_xlabel("Frame",color=C["t"],fontsize=8)

    # Steering
    ax6 = fig.add_subplot(gs[1,4]); sax(ax6,"Steering Angle")
    sl=[d.get("steer",0) for d in det_log]
    if sl:
        ax6.fill_between(F,sl,alpha=.20,color=C["y"])
        ax6.plot(F,sl,color=C["y"],lw=1.2)
        ax6.axhline(0,color=C["t"],ls="--",lw=.6,alpha=.4)
        ax6.set_ylim(-1.2,1.2)
    ax6.yaxis.grid(True,color=C["gr"],alpha=.4)

    # Speed histogram
    ax7 = fig.add_subplot(gs[2,:2]); sax(ax7,"Speed Distribution")
    if spd_log:
        n,bins,patches=ax7.hist(spd_log,bins=30,edgecolor=C["bg"],alpha=.85)
        for patch,left in zip(patches,bins[:-1]):
            patch.set_facecolor(C["r"] if left<10 else C["o"] if left<30 else C["g"])
    ax7.set_xlabel("Speed (km/h)",color=C["t"],fontsize=8)
    ax7.yaxis.grid(True,color=C["gr"],alpha=.4)

    # Risk histogram
    ax8 = fig.add_subplot(gs[2,2:4]); sax(ax8,"Risk Score Distribution")
    if risk_log:
        n,bins,patches=ax8.hist(risk_log,bins=25,edgecolor=C["bg"],alpha=.85)
        for patch,left in zip(patches,bins[:-1]):
            patch.set_facecolor(C["r"] if left>.7 else C["o"] if left>.4 else C["g"])
    ax8.set_xlabel("Risk Score",color=C["t"],fontsize=8)
    ax8.yaxis.grid(True,color=C["gr"],alpha=.4)

    # G-force scatter (simulated from speed changes)
    ax9 = fig.add_subplot(gs[2,4]); sax(ax9,"G-Force Map")
    if det_log:
        gl = [d.get("accel_g",0) for d in det_log]
        tl = [d.get("lat_g",0)   for d in det_log]
        scatter_c = [C["r"] if abs(g)>.4 else C["o"] if abs(g)>.2 else C["g"]
                     for g in gl]
        ax9.scatter(tl,gl,c=scatter_c,s=4,alpha=.6)
        ax9.axhline(0,color=C["t"],lw=.5,alpha=.3)
        ax9.axvline(0,color=C["t"],lw=.5,alpha=.3)
        ax9.set_xlim(-1.2,1.2); ax9.set_ylim(-1.2,1.2)
    ax9.set_xlabel("Lateral G",color=C["t"],fontsize=8)
    ax9.set_ylabel("Long G",color=C["t"],fontsize=8)
    ax9.yaxis.grid(True,color=C["gr"],alpha=.3)
    ax9.xaxis.grid(True,color=C["gr"],alpha=.3)

    plt.savefig(path,dpi=150,bbox_inches="tight",facecolor="#05080f")
    plt.close()
    print(f"  [OK] Dashboard → {path}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def run():
    import shutil

    inp = find_video(INPUT_VIDEO)
    if not inp:
        print("\n[ERROR] No input video found!")
        print("  Place your dashcam video in the same folder.")
        print("  Accepted names: driving_input.mp4, dashcam.mp4, road.mp4, video.mp4 …")
        sys.exit(1)
    print(f"\n[VIDEO] Using: {inp}")

    print("[YOLO] Loading YOLOv8n …")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("  [OK] YOLOv8n ready")
    except Exception as e:
        print(f"  [ERROR] YOLOv8 failed: {e}"); sys.exit(1)

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {inp}"); sys.exit(1)

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO] {W}×{H}  {fps:.1f}fps  {total} frames")

    os.makedirs("driving_output", exist_ok=True)
    out_path = os.path.join("driving_output", OUTPUT_VIDEO)
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                               fps, (W + HUD_W, H))

    perc = Perception()
    eng  = DecisionEngine()
    hud  = HUD(W, H, hw=HUD_W)
    ego  = Ego(speed=35.0)

    spd_log=[]; risk_log=[]; lane_log=[]; dec_log=[]; det_log=[]
    prev_speed = 35.0

    print("[PROCESS] Running … (Ctrl+C to stop early)")
    for fi in tqdm(range(total), desc="  Frames", ncols=70):
        ret, frame = cap.read()
        if not ret: break

        # lane offset (no drawing)
        lane_off  = perc.lane_offset_estimate(frame)
        # crude lane confidence from edge density
        gray_   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges_  = cv2.Canny(cv2.GaussianBlur(gray_,(7,7),0), 40, 120)
        roi_y_  = int(H*0.6)
        lane_conf = min(1.0, edges_[roi_y_:,:].sum() / (W*(H-roi_y_)*0.12))

        # YOLO detection
        raw = model(frame, conf=0.28, iou=0.45,
                    classes=list(DET_CLASSES.keys()), verbose=False)[0]
        dets: List[Det] = []
        for box in raw.boxes:
            cls_  = int(box.cls[0])
            conf_ = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            info = DET_CLASSES[cls_]
            dist = perc.distance((x1,y1,x2,y2), cls_)
            dets.append(Det(cls=cls_, label=info["name"],
                            bbox=(x1,y1,x2,y2), conf=conf_,
                            dist=dist, risk=info["risk"],
                            color=info["color"]))

        cr  = perc.collision_risk(dets, W, H)
        dec = eng.decide(dets, lane_off, cr, lane_conf)

        # compute g-forces
        speed_delta  = eng.speed - prev_speed          # km/h per frame
        accel_g      = np.clip(speed_delta / 3.0, -1.0, 1.0)
        lateral_g    = np.clip(-dec.steer * 0.6, -1.0, 1.0)
        prev_speed   = eng.speed

        ego.speed    = eng.speed
        ego.steer    = dec.steer
        ego.throttle = dec.throttle
        ego.brake    = dec.brake
        ego.accel_g  = float(accel_g)
        ego.lateral_g= float(lateral_g)

        rendered = hud.render(frame, dets, lane_off, lane_conf,
                              dec, ego, cr, fi, total)
        writer.write(rendered)

        spd_log.append(ego.speed)
        risk_log.append(cr)
        lane_log.append(lane_conf)
        dec_log.append(dec.action)
        peds = sum(1 for d in dets if d.label=="Pedestrian")
        cars = sum(1 for d in dets if d.label in ("Car","Truck","Bus"))
        det_log.append({
            "peds":peds,"cars":cars,"total":len(dets),
            "steer":dec.steer,"accel_g":float(accel_g),"lat_g":float(lateral_g)
        })

    cap.release(); writer.release()
    shutil.copy(out_path, OUTPUT_VIDEO)
    print(f"\n  [OK] Video  → {OUTPUT_VIDEO}")

    dash_path = os.path.join("driving_output", DASHBOARD)
    make_dashboard(spd_log, risk_log, lane_log, dec_log, det_log, dash_path)
    shutil.copy(dash_path, DASHBOARD)

    print("\n" + "━"*62)
    print("  COMPLETE")
    print(f"  Video     → {OUTPUT_VIDEO}")
    print(f"  Dashboard → {DASHBOARD}")
    print("━"*62)


if __name__ == "__main__":
    run()