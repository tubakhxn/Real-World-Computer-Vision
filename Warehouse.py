#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        WAREHOUSE FORKLIFT SAFETY INTELLIGENCE SYSTEM  ── v2.1              ║
║        Dev / Creator  :  tubakhxn                                           ║
║                                                                              ║
║  USAGE:                                                                      ║
║    python warehouse_safety_FINAL.py myvideo.mp4                             ║
║    python warehouse_safety_FINAL.py   (auto-detects video)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import subprocess, sys, os, time, warnings, glob, math, json, random
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import datetime

warnings.filterwarnings("ignore")

_REQUIRED = [
    "opencv-python","numpy","matplotlib","ultralytics",
    "torch","torchvision","Pillow","scipy","tqdm",
]

def _ensure_deps():
    for pkg in _REQUIRED:
        mod = pkg.replace("-","_").split("[")[0]
        try:
            __import__(mod)
        except ImportError:
            print(f"  [AUTO-INSTALL] {pkg} …")
            subprocess.check_call(
                [sys.executable,"-m","pip","install",pkg,"-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("━"*72)
print("  WAREHOUSE FORKLIFT SAFETY INTELLIGENCE SYSTEM  |  tubakhxn  v2.1")
print("━"*72)
print("[INIT] Verifying dependencies …")
_ensure_deps()

import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.spatial.distance import cdist
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    "output_video":     "warehouse_safety_OUTPUT.mp4",
    "output_dashboard": "warehouse_safety_DASHBOARD.png",
    "output_dir":       "warehouse_output",
    "event_log":        "safety_events.json",
    "yolo_model":       "yolov8n.pt",
    "conf_thresh":      0.27,
    "iou_thresh":       0.45,
    "iou_match_thresh": 0.22,
    "max_age_frames":   8,
    "trail_length":     50,
    "dist_high_frac":   0.08,
    "dist_medium_frac": 0.15,
    "dist_low_frac":    0.22,
    "hud_width":        320,
    "heatmap_decay":    0.988,
    "heatmap_radius":   30,
}

_PERSON_CLS   = 0
_VEHICLE_CLS  = {1,2,3,5,7}
_FORKLIFT_COCO = {7}

PAL = {
    "bg":       (10,  12,  18),
    "panel":    (14,  17,  26),
    "accent":   (0,   200, 255),
    "green":    (30,  220, 80),
    "amber":    (0,   165, 255),
    "red":      (0,   50,  255),
    "white":    (240, 240, 245),
    "grey":     (90,  95,  110),
    "forklift": (0,   210, 255),
    "worker":   (50,  240, 120),
    "risk_low":  (30,  220, 80),
    "risk_med":  (0,   165, 255),
    "risk_high": (0,   50,  255),
}

_ZONE_DEFS = [
    {
        "name":        "Forklift Corridor A",
        "poly_frac":   [(0.05,0.30),(0.40,0.30),(0.40,0.85),(0.05,0.85)],
        "color":       (0,80,200),
        "max_workers": 0,
    },
    {
        "name":        "Loading Bay",
        "poly_frac":   [(0.60,0.10),(0.95,0.10),(0.95,0.55),(0.60,0.55)],
        "color":       (0,130,220),
        "max_workers": 1,
    },
]

# ── video finder — CLI arg takes absolute priority ─────────────────────
def find_video() -> str:
    # 1. CLI argument — highest priority
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if os.path.isfile(arg):
            return arg
        print(f"[WARN] CLI arg '{arg}' not found, falling back to auto-detect")

    # 2. Prefer warehouse-named files
    for pattern in ["warehouse.*","forklift.*","factory.*","cctv.*"]:
        for c in glob.glob(pattern):
            if "OUTPUT" not in c.upper() and os.path.isfile(c):
                return c
    # 3. Last resort — any video except OUTPUT
    for pattern in ["*.mp4","*.avi","*.mov","*.mkv"]:
        for c in glob.glob(pattern):
            if "OUTPUT" not in c.upper() and os.path.isfile(c):
                return c
    return ""

# ── codec helper — tries H264, falls back to mp4v ─────────────────────
def get_writer(path, fps, size):
    for fourcc_str in ["avc1","H264","h264","mp4v"]:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            w = cv2.VideoWriter(path, fourcc, fps, size)
            if w.isOpened():
                print(f"  [CODEC] Using {fourcc_str}")
                return w
            w.release()
        except Exception:
            pass
    raise RuntimeError("No working video codec found")

# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BBox:
    x1:int; y1:int; x2:int; y2:int
    @property
    def cx(self)->int: return (self.x1+self.x2)//2
    @property
    def cy(self)->int: return (self.y1+self.y2)//2
    @property
    def w(self)->int:  return self.x2-self.x1
    @property
    def h(self)->int:  return self.y2-self.y1
    @property
    def area(self)->float: return float(self.w*self.h)
    @property
    def center(self)->Tuple[int,int]: return (self.cx,self.cy)
    def iou(self,other:"BBox")->float:
        ix1=max(self.x1,other.x1); iy1=max(self.y1,other.y1)
        ix2=min(self.x2,other.x2); iy2=min(self.y2,other.y2)
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        union=self.area+other.area-inter
        return inter/(union+1e-6)

@dataclass
class RawDet:
    bbox:BBox; conf:float; kind:str

@dataclass
class Track:
    tid:      int
    kind:     str
    bbox:     BBox
    conf:     float
    trail:    deque  = field(default_factory=lambda: deque(maxlen=CFG["trail_length"]))
    missed:   int    = 0
    age:      int    = 0
    velocity: Tuple[float,float] = (0.0,0.0)
    _prev_cx: float  = 0.0
    _prev_cy: float  = 0.0
    in_zone:  bool   = False
    zone_name:str    = ""
    risk:     str    = "NONE"
    risk_dist:float  = 9999.0
    color:    Tuple  = field(default_factory=lambda:(
        random.randint(80,230),random.randint(80,230),random.randint(80,230)))

    def update(self, bbox:BBox, conf:float):
        dx=bbox.cx-self._prev_cx; dy=bbox.cy-self._prev_cy
        alpha=0.35
        self.velocity=(alpha*dx+(1-alpha)*self.velocity[0],
                       alpha*dy+(1-alpha)*self.velocity[1])
        self._prev_cx=float(bbox.cx); self._prev_cy=float(bbox.cy)
        self.bbox=bbox; self.conf=conf
        self.trail.append((bbox.cx,bbox.cy))
        self.missed=0; self.age+=1

@dataclass
class SafetyEvent:
    frame:int; timestamp:float; kind:str; desc:str
    worker_id:int; vehicle_id:int; distance:float

@dataclass
class FrameStats:
    n_workers:   int   = 0
    n_forklifts: int   = 0
    near_misses: int   = 0
    high_risk:   int   = 0
    medium_risk: int   = 0
    zone_intrusions: int = 0
    risk_score:  float = 0.0
    alerts: List[str]  = field(default_factory=list)

# ══════════════════════════════════════════════════════════════════════════════
class MOTracker:
    def __init__(self):
        self.tracks:Dict[int,Track]={}
        self._nid=0

    def update(self, dets:List[RawDet])->Dict[int,Track]:
        active_ids=list(self.tracks.keys())
        matched_dets:Set[int]=set()
        for tid in active_ids:
            tr=self.tracks[tid]
            best_iou,best_di=0.0,None
            for di,det in enumerate(dets):
                if det.kind!=tr.kind: continue
                score=tr.bbox.iou(det.bbox)
                if score>best_iou: best_iou,best_di=score,di
            if best_iou>=CFG["iou_match_thresh"] and best_di is not None:
                tr.update(dets[best_di].bbox,dets[best_di].conf)
                matched_dets.add(best_di)
            else:
                tr.missed+=1; tr.trail.append(None)
                if tr.missed>CFG["max_age_frames"]:
                    del self.tracks[tid]
        for di,det in enumerate(dets):
            if di not in matched_dets:
                t=Track(tid=self._nid,kind=det.kind,bbox=det.bbox,conf=det.conf)
                t._prev_cx=float(det.bbox.cx); t._prev_cy=float(det.bbox.cy)
                t.trail.append((det.bbox.cx,det.bbox.cy))
                self.tracks[self._nid]=t; self._nid+=1
        return self.tracks

# ══════════════════════════════════════════════════════════════════════════════
class SafetyAnalyser:
    def __init__(self, frame_w:int, frame_h:int):
        diag=math.hypot(frame_w,frame_h)
        self.dist_high   = diag*CFG["dist_high_frac"]
        self.dist_medium = diag*CFG["dist_medium_frac"]
        self.dist_low    = diag*CFG["dist_low_frac"]
        self.zones=[]
        for zd in _ZONE_DEFS:
            pts=[(int(fx*frame_w),int(fy*frame_h)) for fx,fy in zd["poly_frac"]]
            self.zones.append({**zd,"poly":pts})
        self.events:List[SafetyEvent]=[]
        self.cumulative_nm=0; self.cumulative_high=0
        self.zone_counters:Dict[str,int]=defaultdict(int)
        self._prev_pairs:Set[Tuple[int,int]]=set()

    @staticmethod
    def _pip(px:int,py:int,poly:List[Tuple[int,int]])->bool:
        n=len(poly); inside=False; j=n-1
        for i in range(n):
            xi,yi=poly[i]; xj,yj=poly[j]
            if ((yi>py)!=(yj>py)) and (px<(xj-xi)*(py-yi)/(yj-yi+1e-9)+xi):
                inside=not inside
            j=i
        return inside

    def analyse(self, tracks:Dict[int,Track], frame_idx:int, t_sec:float)->FrameStats:
        st=FrameStats()
        workers  =[tr for tr in tracks.values() if tr.kind=="worker"]
        forklifts=[tr for tr in tracks.values() if tr.kind=="forklift"]
        st.n_workers=len(workers); st.n_forklifts=len(forklifts)
        for tr in tracks.values():
            tr.risk="NONE"; tr.risk_dist=9999.0; tr.in_zone=False; tr.zone_name=""

        current_pairs:Set[Tuple[int,int]]=set()
        for fl in forklifts:
            for wk in workers:
                dist=math.hypot(fl.bbox.cx-wk.bbox.cx,fl.bbox.cy-wk.bbox.cy)
                if dist<fl.risk_dist: fl.risk_dist=dist
                if dist<wk.risk_dist: wk.risk_dist=dist
                if dist<self.dist_high:
                    risk="HIGH"; st.high_risk+=1
                    current_pairs.add((fl.tid,wk.tid))
                    if (fl.tid,wk.tid) not in self._prev_pairs:
                        st.near_misses+=1; self.cumulative_nm+=1; self.cumulative_high+=1
                        self.events.append(SafetyEvent(frame_idx,t_sec,"HIGH_RISK",
                            f"Worker ID{wk.tid} within {dist:.0f}px of Forklift ID{fl.tid}",
                            wk.tid,fl.tid,dist))
                        st.alerts.append(f"HIGH RISK  FL{fl.tid}<>W{wk.tid}  {dist:.0f}px")
                elif dist<self.dist_medium:
                    risk="MEDIUM"; st.medium_risk+=1
                    if (fl.tid,wk.tid) not in self._prev_pairs:
                        self.events.append(SafetyEvent(frame_idx,t_sec,"NEAR_MISS",
                            f"Worker ID{wk.tid} near Forklift ID{fl.tid}",
                            wk.tid,fl.tid,dist))
                        st.alerts.append(f"MEDIUM RISK  FL{fl.tid}<>W{wk.tid}")
                elif dist<self.dist_low:
                    risk="LOW"
                else:
                    continue
                _order={"HIGH":3,"MEDIUM":2,"LOW":1,"NONE":0}
                if _order[risk]>_order[fl.risk]: fl.risk=risk
                if _order[risk]>_order[wk.risk]: wk.risk=risk

        self._prev_pairs=current_pairs
        for zone in self.zones:
            cnt=0
            for wk in workers:
                if self._pip(wk.bbox.cx,wk.bbox.cy,zone["poly"]):
                    wk.in_zone=True; wk.zone_name=zone["name"]; cnt+=1
            self.zone_counters[zone["name"]]+=cnt
            if cnt>zone["max_workers"]:
                st.zone_intrusions+=1
                st.alerts.append(f"ZONE INTRUSION  {zone['name']}  ({cnt} workers)")
                self.events.append(SafetyEvent(frame_idx,t_sec,"ZONE_INTRUSION",
                    f"{cnt} worker(s) in {zone['name']}",-1,-1,0.0))

        st.risk_score=min(100.0,
            st.high_risk*35.0+st.medium_risk*15.0+
            st.zone_intrusions*10.0+st.n_workers*1.5)
        return st

# ══════════════════════════════════════════════════════════════════════════════
class WorkerHeatmap:
    def __init__(self): self._map:Optional[np.ndarray]=None
    def update(self,tracks:Dict[int,Track],shape:Tuple):
        h,w=shape[:2]
        if self._map is None: self._map=np.zeros((h,w),np.float32)
        self._map*=CFG["heatmap_decay"]
        for tr in tracks.values():
            if tr.kind=="worker":
                cx,cy=tr.bbox.cx,tr.bbox.cy
                if 0<=cx<w and 0<=cy<h:
                    cv2.circle(self._map,(cx,cy),CFG["heatmap_radius"],1.0,-1)
    def overlay(self,frame:np.ndarray,alpha:float=0.28):
        if self._map is None: return
        norm=cv2.normalize(self._map,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        col=cv2.applyColorMap(norm,cv2.COLORMAP_INFERNO)
        mask=(norm>15).astype(np.float32)/255.0*alpha
        for c in range(3):
            frame[:,:,c]=np.clip(frame[:,:,c]*(1-mask)+col[:,:,c]*mask,0,255).astype(np.uint8)
    @property
    def data(self): return self._map

# ══════════════════════════════════════════════════════════════════════════════
class FrameRenderer:
    def __init__(self,frame_w:int,frame_h:int,hud_w:int,analyser:SafetyAnalyser):
        self.W=frame_w; self.H=frame_h; self.hw=hud_w; self.analyser=analyser
        self._alert_buffer:deque=deque(maxlen=6)

    @staticmethod
    def _txt(img,text,x,y,scale=0.42,color=(220,220,225),bold=False,shadow=True):
        th=2 if bold else 1
        if shadow:
            cv2.putText(img,text,(x+1,y+1),cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),th+1,cv2.LINE_AA)
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,th,cv2.LINE_AA)

    @staticmethod
    def _bar(img,x,y,w,h,val,maxv,color):
        cv2.rectangle(img,(x,y),(x+w,y+h),(25,28,38),-1)
        filled=int(w*min(val,maxv)/max(maxv,1))
        if filled>0: cv2.rectangle(img,(x,y),(x+filled,y+h),color,-1)

    def _draw_zones(self,frame:np.ndarray):
        ov=frame.copy()
        for zone in self.analyser.zones:
            pts=np.array(zone["poly"],np.int32)
            cv2.fillPoly(ov,[pts],zone["color"])
            cv2.polylines(ov,[pts],True,zone["color"],2)
        cv2.addWeighted(ov,0.18,frame,0.82,0,frame)
        for zone in self.analyser.zones:
            cx=int(np.mean([p[0] for p in zone["poly"]]))
            cy=int(np.mean([p[1] for p in zone["poly"]]))
            self._txt(frame,zone["name"],cx-55,cy,0.40,(180,200,255))

    def _draw_risk_circles(self,frame:np.ndarray,tracks:Dict[int,Track]):
        for tr in tracks.values():
            if tr.kind!="forklift": continue
            for dist,col,thick in [
                (self.analyser.dist_high,  PAL["risk_high"],2),
                (self.analyser.dist_medium,PAL["risk_med"], 1),
                (self.analyser.dist_low,   PAL["risk_low"], 1),
            ]:
                cv2.circle(frame,tr.bbox.center,int(dist),col,thick,cv2.LINE_AA)
            self._txt(frame,f"FL{tr.tid}",tr.bbox.cx-15,tr.bbox.y1-20,
                      0.44,PAL["forklift"],bold=True)

    def _draw_trails(self,frame:np.ndarray,tracks:Dict[int,Track]):
        for tr in tracks.values():
            pts=[p for p in tr.trail if p is not None]
            if len(pts)<2: continue
            col=PAL["forklift"] if tr.kind=="forklift" else PAL["worker"]
            for i in range(1,len(pts)):
                alpha=i/len(pts)
                c=tuple(int(v*alpha) for v in col)
                cv2.line(frame,pts[i-1],pts[i],c,2,cv2.LINE_AA)

    def _draw_predicted_path(self,frame:np.ndarray,tracks:Dict[int,Track]):
        for tr in tracks.values():
            if tr.kind!="forklift": continue
            if abs(tr.velocity[0])<0.5 and abs(tr.velocity[1])<0.5: continue
            pts=[(tr.bbox.cx,tr.bbox.cy)]
            cx,cy=float(tr.bbox.cx),float(tr.bbox.cy)
            for _ in range(12):
                cx+=tr.velocity[0]*1.2; cy+=tr.velocity[1]*1.2
                pts.append((int(cx),int(cy)))
            for i in range(len(pts)-1):
                t_=i/len(pts)
                col=(int(0+t_*0),int(165+t_*45),255)
                cv2.line(frame,pts[i],pts[i+1],col,2,cv2.LINE_AA)
            if len(pts)>=2:
                cv2.arrowedLine(frame,pts[-2],pts[-1],(0,220,255),2,tipLength=0.4)

    def _draw_boxes(self,frame:np.ndarray,tracks:Dict[int,Track]):
        risk_colors={"HIGH":PAL["risk_high"],"MEDIUM":PAL["risk_med"],
                     "LOW":PAL["risk_low"],"NONE":None}
        for tr in tracks.values():
            b=tr.bbox
            base_col=PAL["forklift"] if tr.kind=="forklift" else PAL["worker"]
            rc=risk_colors.get(tr.risk)
            col=rc if rc else base_col
            thick=3 if tr.risk in ("HIGH","MEDIUM") else 2
            cv2.rectangle(frame,(b.x1,b.y1),(b.x2,b.y2),col,thick)
            tl=min(14,b.w//4,b.h//4)
            for sx,sy,dx,dy in [(b.x1,b.y1,1,1),(b.x2,b.y1,-1,1),
                                 (b.x1,b.y2,1,-1),(b.x2,b.y2,-1,-1)]:
                cv2.line(frame,(sx,sy),(sx+dx*tl,sy),col,2,cv2.LINE_AA)
                cv2.line(frame,(sx,sy),(sx,sy+dy*tl),col,2,cv2.LINE_AA)
            kind_tag="FL" if tr.kind=="forklift" else "W"
            label=f"{kind_tag}{tr.tid}  {tr.conf:.0%}"
            if tr.risk!="NONE": label+=f"  [{tr.risk}]"
            (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.42,1)
            cv2.rectangle(frame,(b.x1,b.y1-th-8),(b.x1+tw+6,b.y1),(0,0,0),-1)
            self._txt(frame,label,b.x1+3,b.y1-4,0.42,col)
            if tr.in_zone:
                self._txt(frame,f"! {tr.zone_name[:18]}",b.x1,b.y2+14,0.36,(0,80,220))

    def _draw_alert_banner(self,frame:np.ndarray,alerts:List[str]):
        for a in alerts:
            if a not in self._alert_buffer:
                self._alert_buffer.appendleft(a)
        if not self._alert_buffer: return
        bh=28*len(self._alert_buffer)+10
        cv2.rectangle(frame,(0,0),(self.W-self.hw,bh),(0,0,0),-1)
        cv2.line(frame,(0,bh),(self.W-self.hw,bh),(0,50,180),2)
        for i,alert in enumerate(self._alert_buffer):
            col=(PAL["risk_high"] if "HIGH" in alert else
                 PAL["risk_med"]  if "MEDIUM" in alert or "ZONE" in alert else
                 PAL["risk_low"])
            self._txt(frame,alert,8,22+i*28,0.46,col,bold=True)

    def _draw_hud(self,panel:np.ndarray,st:FrameStats,
                  cum_nm:int,cum_high:int,n_events:int,
                  frame_idx:int,total:int):
        panel[:]=PAL["bg"]
        cv2.line(panel,(0,0),(0,self.H),PAL["accent"],3)
        hw=self.hw

        def t(text,x,y,sc=0.42,col=PAL["white"],bold=False):
            self._txt(panel,text,x,y,sc,col,bold,shadow=False)

        cv2.rectangle(panel,(0,0),(hw,52),PAL["panel"],-1)
        t("SAFETY INTELLIGENCE",8,22,0.52,PAL["accent"],True)
        t("tubakhxn  |  WFSIS v2.1",8,42,0.32,PAL["grey"])
        cv2.line(panel,(4,52),(hw-4,52),PAL["accent"],1)

        t("LIVE DETECTIONS",8,72,0.44,PAL["accent"])
        t(f"Workers   : {st.n_workers}",   10,94, 0.46,PAL["worker"])
        t(f"Forklifts : {st.n_forklifts}", 10,116,0.46,PAL["forklift"])
        cv2.line(panel,(4,126),(hw-4,126),(30,34,48),1)

        t("RISK SCORE",8,144,0.44,PAL["accent"])
        rs=st.risk_score
        rc=(PAL["risk_high"] if rs>60 else PAL["risk_med"] if rs>25 else PAL["risk_low"])
        self._bar(panel,8,148,hw-16,16,rs,100,rc)
        t(f"{rs:.0f}/100",hw-60,162,0.40,rc)
        cv2.line(panel,(4,172),(hw-4,172),(30,34,48),1)

        t("FRAME EVENTS",8,190,0.44,PAL["accent"])
        items=[
            (f"High Risk Events  : {st.high_risk}",   PAL["risk_high"]),
            (f"Medium Risk       : {st.medium_risk}",  PAL["risk_med"]),
            (f"Zone Intrusions   : {st.zone_intrusions}",(0,80,220)),
            (f"Active Alerts     : {len(st.alerts)}",  PAL["amber"]),
        ]
        for i,(txt,col) in enumerate(items):
            t(txt,10,210+i*20,0.42,col)
        cv2.line(panel,(4,296),(hw-4,296),(30,34,48),1)

        t("CUMULATIVE SAFETY LOG",8,314,0.44,PAL["accent"])
        t(f"Near-Misses    : {cum_nm}",   10,336,0.42,PAL["amber"])
        t(f"High-Risk Evts : {cum_high}", 10,358,0.42,PAL["risk_high"])
        t(f"Total Events   : {n_events}", 10,380,0.42,PAL["grey"])
        cv2.line(panel,(4,392),(hw-4,392),(30,34,48),1)

        occ_total=max(1,st.n_workers+st.n_forklifts)
        occ_level=("HIGH" if occ_total>8 else "MEDIUM" if occ_total>3 else "LOW")
        occ_col=(PAL["risk_high"] if occ_level=="HIGH" else
                 PAL["risk_med"]  if occ_level=="MEDIUM" else PAL["risk_low"])
        t("OCCUPANCY LEVEL",8,410,0.44,PAL["accent"])
        cv2.rectangle(panel,(8,416),(hw-8,434),(25,28,38),-1)
        cv2.rectangle(panel,(8,416),(hw-8,434),occ_col,2)
        t(f"  {occ_level}  ({occ_total} entities)",12,430,0.46,occ_col,True)
        cv2.line(panel,(4,444),(hw-4,444),(30,34,48),1)

        t("ACTIVE ALERTS",8,462,0.44,PAL["accent"])
        if st.alerts:
            for i,a in enumerate(st.alerts[:5]):
                col=(PAL["risk_high"] if "HIGH" in a else
                     PAL["risk_med"]  if "MEDIUM" in a or "ZONE" in a else PAL["risk_low"])
                t(f"* {a[:36]}",8,482+i*18,0.36,col)
        else:
            t("  No active alerts",8,482,0.40,PAL["grey"])

        cv2.line(panel,(4,self.H-34),(hw-4,self.H-34),(30,34,48),1)
        prog=frame_idx/max(1,total)
        cv2.rectangle(panel,(4,self.H-30),(hw-4,self.H-16),(25,28,38),-1)
        cv2.rectangle(panel,(4,self.H-30),(int(4+(hw-8)*prog),self.H-16),PAL["accent"],-1)
        t(f"Frame {frame_idx}/{total}  {prog*100:.1f}%",8,self.H-4,0.34,PAL["grey"])

    def render(self,frame:np.ndarray,tracks:Dict[int,Track],st:FrameStats,
               heatmap:WorkerHeatmap,cum_nm:int,cum_high:int,n_events:int,
               frame_idx:int,total:int)->np.ndarray:
        H,W=frame.shape[:2]
        hw=self.hw
        canvas=np.zeros((H,W+hw,3),np.uint8)
        canvas[:,:W]=frame
        main=canvas[:,:W]
        self._draw_zones(main)
        heatmap.overlay(main,alpha=0.28)
        self._draw_trails(main,tracks)
        self._draw_predicted_path(main,tracks)
        self._draw_risk_circles(main,tracks)
        self._draw_boxes(main,tracks)
        self._draw_alert_banner(main,st.alerts)
        cv2.line(canvas,(W,0),(W,H),PAL["accent"],2)
        hud_panel=canvas[:,W:]
        self._draw_hud(hud_panel,st,cum_nm,cum_high,n_events,frame_idx,total)
        return canvas

# ══════════════════════════════════════════════════════════════════════════════
def generate_dashboard(history:List[Dict],heatmap_data:Optional[np.ndarray],
                       events:List[SafetyEvent],zone_counters:Dict[str,int],
                       output_path:str):
    print("\n[DASHBOARD] Generating analytics dashboard …")
    from collections import Counter
    frames=list(range(len(history)))
    workers_t  =[h["n_workers"]   for h in history]
    forklifts_t=[h["n_forklifts"] for h in history]
    risk_t     =[h["risk_score"]  for h in history]
    high_t     =[h["high_risk"]   for h in history]
    medium_t   =[h["medium_risk"] for h in history]
    event_kinds=Counter(e.kind for e in events)

    BG=   "#07090f"; PANEL="#0d1018"; ACC="#00c8ff"; GRN="#1ddd6a"
    AMB=  "#ff9500"; RED= "#ff2d2d";  TXT="#c8d0dc"; GRID="#1a1e2a"

    fig=plt.figure(figsize=(22,12),facecolor=BG)
    fig.suptitle("WAREHOUSE FORKLIFT SAFETY INTELLIGENCE  ──  ANALYTICS DASHBOARD",
                 color=ACC,fontsize=18,fontweight="bold",y=0.98,fontfamily="monospace")
    fig.text(0.985,0.975,"tubakhxn  |  WFSIS v2.1",color="#3a3e50",fontsize=9,
             ha="right",va="top",fontfamily="monospace")
    gs=fig.add_gridspec(3,4,hspace=0.48,wspace=0.38,left=0.05,right=0.97,top=0.93,bottom=0.05)

    def sax(ax,title=""):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.tick_params(colors=TXT,labelsize=8)
        if title: ax.set_title(title,color=ACC,fontsize=10,fontfamily="monospace",pad=7)

    ax1=fig.add_subplot(gs[0,:2]); sax(ax1,"Detected Entities Over Time")
    if frames:
        ax1.fill_between(frames,workers_t,alpha=.22,color=GRN)
        ax1.fill_between(frames,forklifts_t,alpha=.22,color=ACC)
        ax1.plot(frames,workers_t,color=GRN,lw=1.6,label="Workers")
        ax1.plot(frames,forklifts_t,color=ACC,lw=1.6,label="Forklifts")
    ax1.legend(facecolor=PANEL,edgecolor=GRID,labelcolor=TXT,fontsize=9)
    ax1.set_xlabel("Frame",color=TXT,fontsize=8)
    ax1.yaxis.grid(True,color=GRID,alpha=.6)

    ax2=fig.add_subplot(gs[0,2:]); sax(ax2,"Real-Time Risk Score (0–100)")
    if frames:
        ax2.fill_between(frames,risk_t,alpha=.20,color=RED)
        ax2.plot(frames,risk_t,color=RED,lw=1.5)
        ax2.axhline(60,color=RED,ls="--",lw=1,alpha=.7,label="High Risk")
        ax2.axhline(25,color=AMB,ls="--",lw=1,alpha=.7,label="Medium Risk")
        ax2.set_ylim(0,105)
    ax2.legend(facecolor=PANEL,edgecolor=GRID,labelcolor=TXT,fontsize=9)
    ax2.yaxis.grid(True,color=GRID,alpha=.6)

    ax3=fig.add_subplot(gs[1,:2]); sax(ax3,"Risk Events Per Frame")
    if frames:
        ax3.stackplot(frames,high_t,medium_t,colors=[RED,AMB],alpha=.75,
                      labels=["HIGH RISK","MEDIUM RISK"])
    ax3.legend(facecolor=PANEL,edgecolor=GRID,labelcolor=TXT,fontsize=9)
    ax3.yaxis.grid(True,color=GRID,alpha=.6)

    ax4=fig.add_subplot(gs[1,2]); sax(ax4,"Safety Event Types")
    if event_kinds:
        labels_=list(event_kinds.keys()); vals_=list(event_kinds.values())
        ecols=[RED if "HIGH" in l else AMB if "NEAR" in l or "ZONE" in l else GRN for l in labels_]
        ax4.pie(vals_,labels=labels_,colors=ecols,autopct="%1.0f%%",startangle=90,
                textprops={"color":TXT,"fontsize":8},
                wedgeprops={"edgecolor":BG,"linewidth":2})
    else:
        ax4.text(0.5,0.5,"No events",color=TXT,ha="center",va="center",transform=ax4.transAxes)

    ax5=fig.add_subplot(gs[1,3])
    ax5.set_facecolor(PANEL)
    for sp in ax5.spines.values(): sp.set_visible(False)
    ax5.set_xlim(0,1); ax5.set_ylim(0,1); ax5.set_xticks([]); ax5.set_yticks([])
    ax5.set_title("Safety KPIs",color=ACC,fontsize=10,fontfamily="monospace",pad=7)
    avg_workers=np.mean(workers_t) if workers_t else 0
    peak_risk=max(risk_t) if risk_t else 0
    kpis=[("Avg Workers / Frame",f"{avg_workers:.1f}",GRN),
          ("Peak Risk Score",f"{peak_risk:.0f} / 100",RED),
          ("Total High-Risk",str(sum(high_t)),RED),
          ("Total Safety Events",str(len(events)),AMB)]
    for i,(lbl,val,col) in enumerate(kpis):
        y_=0.84-i*0.22
        ax5.add_patch(FancyBboxPatch((0.05,y_-0.09),0.90,0.18,boxstyle="round,pad=0.02",
                      facecolor=BG,edgecolor=col,linewidth=1.6))
        ax5.text(0.50,y_+0.04,val,color=col,fontsize=15,ha="center",va="center",
                fontfamily="monospace",fontweight="bold")
        ax5.text(0.50,y_-0.03,lbl,color=TXT,fontsize=7,ha="center",va="center")

    ax6=fig.add_subplot(gs[2,:2]); sax(ax6,"Worker Activity Heatmap")
    if heatmap_data is not None:
        hm=cv2.resize(heatmap_data,(480,270))
        ax6.imshow(hm,cmap="inferno",interpolation="bilinear",aspect="auto")
        ax6.set_xticks([]); ax6.set_yticks([])
    else:
        ax6.text(0.5,0.5,"Insufficient heatmap data",color=TXT,ha="center",va="center",
                 transform=ax6.transAxes)

    ax7=fig.add_subplot(gs[2,2]); sax(ax7,"Zone Intrusion Counts")
    if zone_counters:
        znames=list(zone_counters.keys()); zcnts=list(zone_counters.values())
        zcols=[RED if c>10 else AMB for c in zcnts]
        bars=ax7.barh(znames,zcnts,color=zcols,height=0.5)
        for bar,val in zip(bars,zcnts):
            ax7.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,
                     str(val),va="center",color=TXT,fontsize=8)
        ax7.set_xlabel("Worker-Frames in Zone",color=TXT,fontsize=8)
    ax7.yaxis.grid(True,color=GRID,alpha=.6)

    ax8=fig.add_subplot(gs[2,3]); sax(ax8,"Risk Score Distribution")
    if risk_t:
        n,bins,patches=ax8.hist(risk_t,bins=20,color=ACC,edgecolor=BG,alpha=.85)
        for patch,left in zip(patches,bins[:-1]):
            patch.set_facecolor(RED if left>60 else AMB if left>25 else GRN)
    ax8.set_xlabel("Risk Score",color=TXT,fontsize=8)
    ax8.yaxis.grid(True,color=GRID,alpha=.6)

    plt.savefig(output_path,dpi=150,bbox_inches="tight",facecolor=BG)
    plt.close()
    print(f"  [OK] Dashboard → {output_path}")

# ══════════════════════════════════════════════════════════════════════════════
def run():
    import shutil

    inp=find_video()
    if not inp:
        print("\n[ERROR] No input video found!")
        print("  Usage: python warehouse_safety_FINAL.py your_video.mp4")
        print("  Or name your file: warehouse.mp4")
        sys.exit(1)
    print(f"\n[VIDEO] Using: {inp}")

    print("[YOLO] Loading YOLOv8n …")
    try:
        from ultralytics import YOLO
        model=YOLO(CFG["yolo_model"])
        print("  [OK] YOLOv8n ready")
    except Exception as e:
        print(f"  [ERROR] YOLOv8 failed: {e}"); sys.exit(1)

    cap=cv2.VideoCapture(inp)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {inp}"); sys.exit(1)

    W    =int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  =cap.get(cv2.CAP_PROP_FPS) or 25.0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO] {W}×{H}  {fps:.1f}fps  {total} frames  ({total/fps:.1f}s)")

    os.makedirs(CFG["output_dir"],exist_ok=True)
    out_path=os.path.join(CFG["output_dir"],CFG["output_video"])
    hw=CFG["hud_width"]

    try:
        writer=get_writer(out_path,fps,(W+hw,H))
    except RuntimeError as e:
        print(f"[ERROR] {e}"); sys.exit(1)

    tracker =MOTracker()
    analyser=SafetyAnalyser(W,H)
    heatmap =WorkerHeatmap()
    renderer=FrameRenderer(W,H,hw,analyser)

    history:List[Dict]=[]
    log_path=os.path.join(CFG["output_dir"],CFG["event_log"])

    print(f"\n[PROCESS] Analysing {total} frames …")
    t0=time.time()

    for fi in tqdm(range(total),desc="  Frames",ncols=72,unit="fr"):
        ret,frame=cap.read()
        if not ret: break

        results=model(frame,conf=CFG["conf_thresh"],iou=CFG["iou_thresh"],
                      classes=[_PERSON_CLS,*_VEHICLE_CLS],verbose=False)[0]

        raw_dets:List[RawDet]=[]
        for box in results.boxes:
            cls =int(box.cls[0]); conf=float(box.conf[0])
            x1,y1,x2,y2=map(int,box.xyxy[0])
            bbox=BBox(x1,y1,x2,y2)
            if cls==_PERSON_CLS:
                kind="worker"
            elif cls in _FORKLIFT_COCO:
                ar=bbox.w/max(bbox.h,1)
                kind="forklift" if ar<1.8 else "worker"
            else:
                kind="forklift"
            raw_dets.append(RawDet(bbox=bbox,conf=conf,kind=kind))

        tracks=tracker.update(raw_dets)
        t_sec=fi/fps
        st=analyser.analyse(tracks,fi,t_sec)
        heatmap.update(tracks,frame.shape)

        canvas=renderer.render(frame,tracks,st,heatmap,
                               analyser.cumulative_nm,analyser.cumulative_high,
                               len(analyser.events),fi,total)
        writer.write(canvas)

        history.append({"n_workers":st.n_workers,"n_forklifts":st.n_forklifts,
                        "risk_score":st.risk_score,"high_risk":st.high_risk,
                        "medium_risk":st.medium_risk,"zone_intrusions":st.zone_intrusions})

    elapsed=time.time()-t0
    cap.release(); writer.release()
    shutil.copy(out_path,CFG["output_video"])
    print(f"\n  [OK] Video → {CFG['output_video']}")
    print(f"       Processed {len(history)} frames in {elapsed:.1f}s  "
          f"({len(history)/max(elapsed,1e-6):.1f} fps)")

    log_data=[{"frame":e.frame,"time_sec":round(e.timestamp,3),"type":e.kind,
               "description":e.desc,"worker_id":e.worker_id,
               "vehicle_id":e.vehicle_id,"distance_px":round(e.distance,1)}
              for e in analyser.events]
    with open(log_path,"w") as fh:
        json.dump(log_data,fh,indent=2)
    shutil.copy(log_path,CFG["event_log"])
    print(f"       Event log → {CFG['event_log']}  ({len(log_data)} events)")

    dash_path=os.path.join(CFG["output_dir"],CFG["output_dashboard"])
    generate_dashboard(history,heatmap.data,analyser.events,
                       dict(analyser.zone_counters),dash_path)
    shutil.copy(dash_path,CFG["output_dashboard"])

    print("\n" + "━"*72)
    print("  PIPELINE COMPLETE")
    print(f"  Output video   → {CFG['output_video']}")
    print(f"  Dashboard      → {CFG['output_dashboard']}")
    print(f"  Event log      → {CFG['event_log']}")
    print(f"  Near-misses    :  {analyser.cumulative_nm}")
    print(f"  High-risk evts :  {analyser.cumulative_high}")
    print(f"  Total events   :  {len(analyser.events)}")
    print("━"*72 + "\n")

if __name__=="__main__":
    run()