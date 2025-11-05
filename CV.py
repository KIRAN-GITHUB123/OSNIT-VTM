from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock
import argparse, json, hashlib, logging, os, math, cv2, numpy as np, requests
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set, ClassVar
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from tqdm import tqdm
import collections
import folium
import pytesseract
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    KeepInFrame, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from PIL import Image 
TESSERACT_LOCK = Lock()

# ----------------------------------------------------------------------
# DL (MPS) – safe
# ----------------------------------------------------------------------
YOLO_MODEL = None
CLIP_MODEL = None
CLIP_PREPROCESS = None
DEVICE = "cpu"

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception: YOLO_AVAILABLE = False

try:
    import torch, clip 
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    CLIP_AVAILABLE = True
except Exception: CLIP_AVAILABLE = False

# ----------------------------------------------------------------------
# CONFIG – TUNED FOR BALANCED PRECISION
# ----------------------------------------------------------------------
class Config:
    # --- Paths & General ---
    OUTPUT   = Path("osint_vtm_v23")
    RAW      = OUTPUT / "raw"
    SNIP     = OUTPUT / "snippets"
    CACHE    = OUTPUT / "cache"
    MODELS   = OUTPUT / "models"
    PROC_W   = 1024  # Processing width
    WORKERS  = max(1, os.cpu_count() // 2)

    # --- Thresholds ---
    DL_BOOST_MAX   = 0.06  # Max confidence boost from DL verification
    NMS_IOU        = 0.40  # Non-Maximal Suppression IoU
    DL_IOU_THRESH  = 0.15  # IoU needed between Classical & YOLO box to get boost
    YOLO_CONF      = 0.25  # YOLO confidence threshold
    CLIP_CONF      = 0.23  # CLIP similarity threshold
    
    # --- v23.4 Balanced Thresholds ---
    CLASSICAL_TH_VERIFIED = 0.72
    CLASSICAL_TH_UNVERIFIED = 0.95

    # --- Lenient Thresholds for 2nd Pass ---
    LENIENT_TH_VERIFIED = 0.65
    LENIENT_TH_UNVERIFIED = 0.75 

    # --- Detector Tuning (Classical) ---
    # L10: Camera (Hough Circles)
    CAM_DP         = 1.2
    CAM_MIN_DIST   = 40
    CAM_PARAM1     = 60
    CAM_PARAM2     = 32
    CAM_MIN_R      = 8
    CAM_MAX_R      = 38
    CAM_HOUSING_R  = 18
    CAM_LENS_RATIO = 0.75
    CAM_HOUSING_STD_MAX = 35.0

    # L14: Entry (Contour + Corner)
    ENTRY_MIN_AREA     = 3000
    ENTRY_MAX_AREA_R   = 0.45
    ENTRY_MIN_H        = 50
    ENTRY_MIN_W        = 30
    ENTRY_ASP_MIN      = 1.7
    ENTRY_ASP_MAX      = 3.5
    ENTRY_SOL_MIN      = 0.74
    ENTRY_CORNER_TH    = 0.04
    ENTRY_CORNER_MIN   = 2
    ENTRY_CORNER_MAX   = 55
    ENTRY_PANEL_STD_MAX = 45.0

    # L8: Graffiti (Color + Texture + OCR)
    GRAF_MIN_AREA      = 4000
    GRAF_LAP_VAR_MIN   = 95.0
    GRAF_SAT_MEAN_MIN  = 100.0
    GRAF_EDGE_DENS_MIN = 0.15
    GRAF_OCR_MIN_LEN   = 5
    
    # L16: Person (HOG)
    PERSON_WIN_STRIDE = (8,8)
    PERSON_PADDING    = (8,8)
    PERSON_SCALE      = 1.07
    PERSON_FINAL_TH   = 1.0
    PERSON_MIN_CONF   = 0.6  # <-- FIX: Loosened from 0.9
    PERSON_ASP_MIN    = 1.5
    
    # L11: License Plate (Morphology + Contour)
    LP_MIN_AREA       = 1500
    LP_MAX_AREA       = 50000
    LP_ASP_MIN        = 2.0
    LP_ASP_MAX        = 5.5
    LP_EXTENT_MIN     = 0.60
    LP_DENS_MIN       = 0.20
    LP_DENS_MAX       = 0.70

    # L13: NEW Stop Sign (Color + Shape)
    STOP_SIGN_MIN_AREA = 1000
    STOP_SIGN_MAX_AREA = 50000
    STOP_SIGN_ASP_DEV  = 0.2
    STOP_SIGN_SOL_MIN  = 0.85

    # L13: NEW Speed Sign (Shape + Density)
    SPEED_SIGN_MIN_AREA = 900
    SPEED_SIGN_MAX_AREA = 40000
    SPEED_SIGN_ASP_MIN  = 1.2
    SPEED_SIGN_ASP_MAX  = 1.8
    SPEED_SIGN_EXT_MIN  = 0.70
    SPEED_SIGN_DENS_MIN = 0.15
    SPEED_SIGN_DENS_MAX = 0.60

    # L19: Face (Haar Cascade)
    FACE_SCALE_F      = 1.05
    FACE_MIN_NEIGH    = 3  # <-- FIX: Loosened from 4
    FACE_MIN_SIZE_R   = 12
    FACE_MIN_SIZE_ABS = 20
    
    # L36: Tracker (KLT)
    TRACK_MAX_CORNERS = 130
    TRACK_QUAL_LEVEL  = 0.26
    TRACK_MIN_DIST    = 5
    TRACK_MIN_GOOD_FEAT = 16
    TRACK_MOTION_TH   = 2.5

    @classmethod
    def init(cls):
        """Creates output directories and downloads ML models."""
        for p in (cls.OUTPUT, cls.RAW, cls.SNIP, cls.CACHE, cls.MODELS): p.mkdir(exist_ok=True)
        
        global YOLO_MODEL
        yolo_path = cls.MODELS / "yolov8x-worldv2.pt" 
        
        if YOLO_AVAILABLE:
            if not yolo_path.exists():
                logging.critical(f"YOLO model not found at {yolo_path}")
                logging.critical(f"Please move your downloaded 'yolov8x-worldv2.pt' file to that directory.")
                raise FileNotFoundError(f"Model not found: {yolo_path}")
                
            try:
                logging.info(f"Loading local YOLO-World model from {yolo_path}...")
                YOLO_MODEL = YOLO(str(yolo_path)) 
                YOLO_MODEL.fuse() 
                YOLO_MODEL.model.eval()
                logging.info(f"YOLOv8x-World model loaded successfully on {DEVICE}.")
            except Exception as e:
                logging.error(f"Failed to load local YOLO-World model: {e}")
                YOLO_MODEL = None
        
        global CLIP_MODEL, CLIP_PREPROCESS
        if CLIP_AVAILABLE:
            try:
                import torch, clip
                logging.info("Loading CLIP model (ViT-B/32)...")
                CLIP_MODEL, CLIP_PREPROCESS = clip.load('ViT-B/32', device=DEVICE)
                CLIP_MODEL.eval()
                logging.info(f"CLIP model loaded successfully on {DEVICE}.")
            except Exception as e:
                logging.error(f"Failed to load CLIP model: {e}")
                CLIP_MODEL = None

# ----------------------------------------------------------------------
# DATA CLASSES
# ----------------------------------------------------------------------
@dataclass
class Img:
    path: Path
    id: str
    lat: float; lon: float
    ts: Optional[datetime]
    heading: Optional[float]
    data: bytes

@dataclass
class Finding:
    fid: int
    img_path: Path
    typ: str
    score: float
    desc: str
    lat: float; lon: float
    bbox: Tuple[int,int,int,int]  
    snippet_path: Optional[Path]
    classical_conf: float
    dl_boost: float = 0.0
    risk_score: float = 0.0
    mitigation: str = ""
    ts: Optional[datetime] = None

@dataclass
class PreprocessedImage:
    """Holds all pre-computed image maps to avoid redundant work."""
    bgr: np.ndarray
    gray: np.ndarray
    blur: np.ndarray
    hsv: np.ndarray
    edges: np.ndarray
    scale: float
    h: int
    w: int

@dataclass
class DLDetection:
    """Stores results from the single-pass DL run."""
    yolo_boxes: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    WORLD_CLASSES: ClassVar[List[str]] = [
        "person", "face", "security camera", "dome camera", "door", 
        "car", "truck", "bus", "license plate", 
        "stop sign", "speed limit sign"
    ]

    YOLO_CLASS_MAP: ClassVar[Dict[str, Set[str]]] = {
        "camera": {"security camera", "dome camera"},
        "entry": {"door"},
        "person": {"person"},
        "face": {"face", "person"},
        "license_plate": {"license plate", "car", "truck", "bus"},
        "stop_sign": {"stop sign"},
        "speed_sign": {"speed limit sign"}
    }
    
    @classmethod
    def from_yolo(cls, bgr: np.ndarray) -> 'DLDetection':
        if not YOLO_MODEL:
            return cls()
        detections = cls()
        try:
            YOLO_MODEL.set_classes(cls.WORLD_CLASSES)
            
            results = YOLO_MODEL(bgr, imgsz=640, conf=Config.YOLO_CONF, verbose=False, device=DEVICE)
            for r in results:
                for box in r.boxes:
                    cls_name = r.names[int(box.cls)]
                    detections.add(cls_name, box.xyxy.cpu().numpy().squeeze().astype(int))
        except Exception as e:
            logging.warning(f"YOLO-World inference failed: {e}")
        return detections

    def add(self, cls_name: str, box: np.ndarray):
        if cls_name not in self.yolo_boxes:
            self.yolo_boxes[cls_name] = []
        self.yolo_boxes[cls_name].append(box)

# ----------------------------------------------------------------------
# MAPILLARY
# ----------------------------------------------------------------------
MAPILLARY_TOKEN = "" # <<< IMPORTANT: SET YOUR TOKEN HERE
if not MAPILLARY_TOKEN: 
    logging.critical("MAPILLARY_ACCESS_TOKEN is not set. Please edit the script.")
    raise RuntimeError("Set MAPILLARY_ACCESS_TOKEN")

def _cache(url: str) -> bytes:
    """Caches request results to disk."""
    h = hashlib.md5(url.encode()).hexdigest()
    p = Config.CACHE / h
    if p.exists(): return p.read_bytes()
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.content
        p.write_bytes(data)
        return data
    except requests.RequestException as e:
        logging.error(f"Failed to fetch or cache URL {url}: {e}")
        return b""

def fetch_images(lat, lon, radius) -> List[Img]:
    """Fetches image metadata and data from Mapillary."""
    logging.info(f"Fetching images within {radius}m of ({lat}, {lon})...")
    
    try:
        recent_date_str = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%dT%H:%M:%SZ')
    except ValueError:
        recent_date_str = "2020-01-01T00:00:00Z"

    delta = radius / 111_320.0
    bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"
    
    url = (
        f"https://graph.mapillary.com/images?access_token={MAPILLARY_TOKEN}"
        f"&bbox={bbox}"
        f"&fields=id,thumb_2048_url,captured_at,compass_angle,geometry"
        f"&limit=300"
        f"&start_captured_at={recent_date_str}"
    )
    
    meta_data = _cache(url)
    if not meta_data:
        logging.error("Failed to fetch image metadata.")
        return []
        
    try:
        meta = json.loads(meta_data).get("data", [])
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON response from Mapillary.")
        return []

    imgs = []
    for f in meta:
        img_url = f.get("thumb_2048_url")
        if not img_url: continue
        
        raw = _cache(img_url)
        if not raw:
            logging.warning(f"Failed to fetch image data for {f['id']}")
            continue
            
        p = Config.RAW / f"img_{f['id']}.jpg"
        p.write_bytes(raw)
        
        ts = None
        if f.get("captured_at"):
            try:
                ts = datetime.fromtimestamp(f.get("captured_at", 0) / 1000)
            except:
                pass 
                
        imgs.append(Img(
            p, f['id'], f['geometry']['coordinates'][1], f['geometry']['coordinates'][0], 
            ts, f.get("compass_angle"), raw
        ))
    
    if not imgs:
        logging.warning("Fetched 0 *recent* images. Your report may be empty or use cached old data.")
    else:
        logging.info(f"Fetched {len(imgs)} recent images.")
    return imgs

# ----------------------------------------------------------------------
#  DETECTOR
# ----------------------------------------------------------------------
class V23Detector:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier()
        if not self.cascade.load(cascade_path):
            logging.warning(f"Haar cascade not found at {cascade_path}. Face detection disabled.")
            self.cascade = None

    @staticmethod
    def _preprocess(bgr: np.ndarray) -> Optional[PreprocessedImage]:
        """Pre-computes all common image representations."""
        if bgr is None or bgr.size == 0:
            return None
        h0, w0 = bgr.shape[:2]
        if h0 == 0 or w0 == 0:
            return None
            
        scale = Config.PROC_W / w0 if w0 > Config.PROC_W else 1.0
        if scale != 1.0:
            w = Config.PROC_W
            h = int(h0 * scale)
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        else:
            w, h = w0, h0
            
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray_eq, (5,5), 0)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(blur, 50, 150)
        
        return PreprocessedImage(bgr, gray_eq, blur, hsv, edges, scale, h, w)

    def camera(self, gray: np.ndarray, blur: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], float]]:
        # L10: Camera (Hough Circles for Dome Cameras)
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, 
            dp=Config.CAM_DP, minDist=Config.CAM_MIN_DIST, 
            param1=Config.CAM_PARAM1, param2=Config.CAM_PARAM2, 
            minRadius=Config.CAM_MIN_R, maxRadius=Config.CAM_MAX_R
        )
        out = []
        if circles is not None:
            for (cx_f, cy_f, r_f) in circles[0]:
                cx, cy, r = int(cx_f), int(cy_f), int(r_f)

                lens_mask = np.zeros_like(gray); cv2.circle(lens_mask,(cx,cy),r,255,-1)
                housing_mask = np.zeros_like(gray); cv2.circle(housing_mask,(cx,cy),r+Config.CAM_HOUSING_R,255,-1)
                housing_mask = cv2.subtract(housing_mask, lens_mask)
                
                mean_lens = cv2.mean(gray, lens_mask)[0]
                mean_housing, std_housing = cv2.meanStdDev(gray, mask=housing_mask)
                mean_housing = mean_housing[0][0]
                std_housing = std_housing[0][0]

                if mean_lens >= mean_housing * Config.CAM_LENS_RATIO: continue
                if std_housing > Config.CAM_HOUSING_STD_MAX: continue
                
                r_box = r + Config.CAM_HOUSING_R // 2
                bbox = (cx-r_box, cy-r_box, 2*r_box, 2*r_box)
                out.append((bbox, 0.89))
        
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if not (500 < area < 5000): continue
            if not (0.8 < w/h < 1.8): continue
            
            if cv2.mean(gray[y:y+h, x:x+w])[0] < 80:
                out.append(((x,y,w,h), 0.75)) # Lower confidence, relies on DL boost
                
        return out

    def entry(self, gray: np.ndarray, edges: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], float]]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours,_ = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cand = []
        img_area = gray.size
        for c in contours:
            area = cv2.contourArea(c)
            if not (Config.ENTRY_MIN_AREA < area < img_area * Config.ENTRY_MAX_AREA_R): continue
            
            x,y,w,h = cv2.boundingRect(c)
            if h < Config.ENTRY_MIN_H or w < Config.ENTRY_MIN_W: continue
            
            asp = h/(w+1e-6); sol = area/((w*h)+1e-6)
            if not (Config.ENTRY_ASP_MIN < asp < Config.ENTRY_ASP_MAX and sol > Config.ENTRY_SOL_MIN): continue
            
            handle_roi = gray[y+h//3 : y+2*h//3, x+w//4 : x+3*w//4]
            if handle_roi.size == 0: continue
            
            dst = cv2.cornerHarris(np.float32(handle_roi), 2, 3, 0.04)
            corners = np.sum(dst > Config.ENTRY_CORNER_TH * dst.max())
            
            if not (Config.ENTRY_CORNER_MIN <= corners <= Config.ENTRY_CORNER_MAX): continue

            roi_top = gray[y+5:y+h//3-5, x+5:x+w-5]
            roi_bottom = gray[y+2*h//3+5:y+h-5, x+5:x+w-5]
            
            if roi_top.size > 0 and roi_bottom.size > 0:
                panel_roi = np.concatenate((roi_top, roi_bottom))
                if panel_roi.size > 0:
                    _, stddev = cv2.meanStdDev(panel_roi)
                    if stddev[0][0] > Config.ENTRY_PANEL_STD_MAX:
                        continue 
            
            cand.append(((x,y,w,h), 0.91))
        return cand

    def graffiti(self, bgr: np.ndarray, hsv: np.ndarray, gray: np.ndarray, edges: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], float]]:
        s_channel = hsv[:,:,1]
        _, s_thresh = cv2.threshold(s_channel, Config.GRAF_SAT_MEAN_MIN, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((10,10), np.uint8)
        s_thresh = cv2.morphologyEx(s_thresh, cv2.MORPH_CLOSE, kernel)
        
        contours,_ = cv2.findContours(s_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < Config.GRAF_MIN_AREA: continue
            
            x,y,w,h = cv2.boundingRect(c)
            if w <= 0 or h <= 0: continue
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_edges = edges[y:y+h, x:x+w]
            
            lap_var = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
            edge_density = np.count_nonzero(roi_edges) / (w * h)

            if lap_var > Config.GRAF_LAP_VAR_MIN and edge_density > Config.GRAF_EDGE_DENS_MIN:
                try:
                    with TESSERACT_LOCK:
                        _, ocr_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        text = pytesseract.image_to_string(ocr_thresh, config='--psm 11', timeout=5.0)
                    clean_text = ''.join(e for e in text if e.isalnum())
                    
                    if len(clean_text) > Config.GRAF_OCR_MIN_LEN:
                        continue 
                except Exception as e:
                    logging.debug(f"Pytesseract failed, assuming not-text: {e}")
                    pass 
                
                score = 0.86 + min(0.1, lap_var / 5000) + min(0.04, edge_density * 0.1)
                out.append(((x,y,w,h), min(1.0, score)))
        return out

    def person(self, gray: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], float]]:
        try:
            boxes, weights = self.hog.detectMultiScale(
                gray, 
                winStride=Config.PERSON_WIN_STRIDE, 
                padding=Config.PERSON_PADDING, 
                scale=Config.PERSON_SCALE, 
                finalThreshold=Config.PERSON_FINAL_TH
            )
            out = []
            for (x,y,w,h), wt in zip(boxes, weights):
                if h == 0 or w == 0: continue
                # --- USE TUNED THRESHOLD ---
                if wt > Config.PERSON_MIN_CONF and h/w > Config.PERSON_ASP_MIN:
                    score = min(1.0, 0.7 + (wt - Config.PERSON_FINAL_TH) * 0.2) # Base score 0.7
                    out.append(((x,y,w,h), score))
            return out
        except Exception as e: 
            logging.debug(f"HOG detection failed: {e}")
            return []

    def license_plate(self, gray: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], float]]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)) 
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (Config.LP_MIN_AREA < area < Config.LP_MAX_AREA): continue
                
            x, y, w, h = cv2.boundingRect(c)
            if h == 0 or w == 0: continue

            aspect_ratio = w / h
            if not (Config.LP_ASP_MIN < aspect_ratio < Config.LP_ASP_MAX): continue
                
            extent = area / (w * h)
            if extent < Config.LP_EXTENT_MIN: continue
                
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0: continue
            
            _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            non_zero = cv2.countNonZero(roi_thresh) / roi.size
            if not (Config.LP_DENS_MIN < non_zero < Config.LP_DENS_MAX): continue

            conf = 0.82 + (extent - 0.60) * 0.10 
            cand.append(((x,y,w,h), min(0.89, conf)))
        return cand

    def detect_stop_sign(self, hsv: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], float]]:
        mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
        mask = cv2.morphologyEx(mask1 | mask2, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (Config.STOP_SIGN_MIN_AREA < area < Config.STOP_SIGN_MAX_AREA): continue
            
            x,y,w,h = cv2.boundingRect(c)
            if w == 0 or h == 0: continue
            
            asp = w / h
            if abs(asp - 1.0) > Config.STOP_SIGN_ASP_DEV: continue
            
            sol = area / (w * h)
            if sol < Config.STOP_SIGN_SOL_MIN: continue
            
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 8:
                cand.append(((x,y,w,h), 0.92))
        return cand

    def detect_speed_sign(self, gray: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], float]]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        _, thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 11))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (Config.SPEED_SIGN_MIN_AREA < area < Config.SPEED_SIGN_MAX_AREA): continue

            x, y, w, h = cv2.boundingRect(c)
            if h == 0 or w == 0: continue

            aspect_ratio = h / w
            if not (Config.SPEED_SIGN_ASP_MIN < aspect_ratio < Config.SPEED_SIGN_ASP_MAX): continue
            
            extent = area / (w * h)
            if extent < Config.SPEED_SIGN_EXT_MIN: continue

            roi = gray[y:y+h, x:x+w]
            if roi.size == 0: continue
            
            _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            non_zero = cv2.countNonZero(roi_thresh) / roi.size
            if not (Config.SPEED_SIGN_DENS_MIN < non_zero < Config.SPEED_SIGN_DENS_MAX): continue
                
            cand.append(((x,y,w,h), 0.88))
        return cand

    def face(self, gray: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], float]]:
        if self.cascade is None or self.cascade.empty():
            return []
        try:
            h, w = gray.shape
            min_size_val = max(Config.FACE_MIN_SIZE_ABS, min(w, h) // Config.FACE_MIN_SIZE_R)
            min_size = (min_size_val, min_size_val)
            
            faces = self.cascade.detectMultiScale(
                gray,
                scaleFactor=Config.FACE_SCALE_F,
                minNeighbors=Config.FACE_MIN_NEIGH, # Use tuned threshold
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return [((x,y,w,h), 0.90) for (x,y,w,h) in faces]
        except Exception as e:
            logging.debug(f"Face detection failed: {e}")
            return []

    def _verify_dl(self, bgr: np.ndarray, bbox_xywh: Tuple[int,int,int,int], typ: str, dl_dets: DLDetection) -> float:
        """Verifies a classical detection using pre-computed YOLO and live CLIP."""
        if not (YOLO_MODEL or CLIP_MODEL): return 0.0
        
        x,y,w,h = bbox_xywh
        h_img, w_img = bgr.shape[:2]
        
        boost = 0.0
        if YOLO_MODEL:
            yolo_classes = DLDetection.YOLO_CLASS_MAP.get(typ, set())
            if yolo_classes:
                classical_box = np.array([x, y, x+w, y+h])
                best_iou = 0.0
                for cls_name in yolo_classes:
                    for yolo_box_xyxy in dl_dets.yolo_boxes.get(cls_name, []):
                        iou = self._iou(classical_box, yolo_box_xyxy)
                        best_iou = max(best_iou, iou)
                
                if best_iou > Config.DL_IOU_THRESH:
                    boost = max(boost, (best_iou - Config.DL_IOU_THRESH) * 0.1)
        
        if CLIP_MODEL:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x+w), min(h_img, y+h)
            if x1 >= x2 or y1 >= y2: return boost
            
            crop = bgr[y1:y2, x1:x2]
            if crop.size == 0 or min(crop.shape[:2]) < 20: return boost
            
            try:
                img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                img = CLIP_PREPROCESS(img_pil).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    emb = CLIP_MODEL.encode_image(img).float()
                    emb /= emb.norm(dim=-1, keepdim=True)
                
                texts = {
                    "camera":"photo of a security camera dome lens",
                    "entry":"photo of a metal door handle or lock",
                    "graffiti":"photo of spray paint graffiti on a wall",
                    "person":"photo of a pedestrian walking",
                    "license_plate":"photo of a vehicle license plate",
                    "face":"photo of a human face",
                    "stop_sign": "photo of a red stop sign",
                    "speed_sign": "photo of a speed limit sign"
                }
                import clip
                txt = clip.tokenize([texts.get(typ, "object")]).to(DEVICE)
                with torch.no_grad():
                    txt_emb = CLIP_MODEL.encode_text(txt).float()
                    txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
                
                sim = (emb @ txt_emb.T).item()
                if sim > Config.CLIP_CONF:
                    boost = max(boost, (sim - Config.CLIP_CONF) * 0.17)
            except Exception as e:
                logging.debug(f"CLIP verification failed: {e}")
                pass 

        return min(boost, Config.DL_BOOST_MAX)

    @staticmethod
    def _iou(boxA, boxB):
        """Calculates Intersection over Union (IoU) for two xyxy boxes."""
        if boxA is None or boxB is None: return 0.0
        boxA = np.array(boxA).flatten()
        boxB = np.array(boxB).flatten()
        if boxA.shape[0] != 4 or boxB.shape[0] != 4: return 0.0

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        if boxAArea + boxBArea - interArea == 0: return 0.0
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    @staticmethod
    def _nms(boxes_xywh, scores, iou_thr):
        """Performs Non-Maximal Suppression on xywh boxes."""
        if len(boxes_xywh) == 0: return [], []
        
        boxes = np.zeros_like(boxes_xywh)
        boxes[:,0] = boxes_xywh[:,0]
        boxes[:,1] = boxes_xywh[:,1]
        boxes[:,2] = boxes_xywh[:,0] + boxes_xywh[:,2]
        boxes[:,3] = boxes_xywh[:,1] + boxes_xywh[:,3]
        
        x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        area = (x2-x1+1)*(y2-y1+1)
        idxs = np.argsort(scores)[::-1]
        
        keep = []
        while len(idxs)>0:
            i = idxs[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0, xx2-xx1+1)
            h = np.maximum(0, yy2-yy1+1)
            inter = w * h
            
            iou_val = inter / (area[i] + area[idxs[1:]] - inter + 1e-6)
            
            idxs = idxs[1:][iou_val <= iou_thr]
            
        return [boxes_xywh[i] for i in keep], [scores[i] for i in keep]

    def detect(self, img: Img) -> List[Finding]:
        """Runs the full classical+DL pipeline on a single image."""
        try:
            bgr = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
            pre = self._preprocess(bgr)
            if pre is None: 
                return []
        except Exception as e:
            logging.warning(f"Failed to decode or preprocess image {img.id}: {e}")
            return []

        dl_detections = DLDetection.from_yolo(pre.bgr)

        raw = {
            "camera": self.camera(pre.gray, pre.blur),
            "entry": self.entry(pre.gray, pre.edges),
            "graffiti": self.graffiti(pre.bgr, pre.hsv, pre.gray, pre.edges),
            "person": self.person(pre.gray),
            "license_plate": self.license_plate(pre.gray),
            "stop_sign": self.detect_stop_sign(pre.hsv),
            "speed_sign": self.detect_speed_sign(pre.gray),
            "face": self.face(pre.gray)
        }

        findings = []
        fid_counter = 0
        for typ, dets in raw.items():
            if not dets: continue
            
            boxes_xywh = np.array([d[0] for d in dets])
            scores = np.array([d[1] for d in dets])
            
            keep_boxes, keep_scores = self._nms(boxes_xywh, scores, Config.NMS_IOU)

            for (x,y,w,h), conf in zip(keep_boxes, keep_scores):
                
                dl_boost = self._verify_dl(pre.bgr, (x,y,w,h), typ, dl_detections)
                final_score = min(1.0, conf + dl_boost)
                
                if pre.scale != 1.0:
                    x,y,w,h = int(x/pre.scale), int(y/pre.scale), int(w/pre.scale), int(h/pre.scale)

                snippet_path = None
                try:
                    bgr_orig = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_COLOR)
                    h_orig, w_orig = bgr_orig.shape[:2]
                    
                    cx = x + w // 2
                    cy = y + h // 2
                    max_dim = max(w, h)
                    side = int(max_dim * 1.75) 
                    x1 = max(0, cx - side // 2)
                    y1 = max(0, cy - side // 2)
                    x2 = min(w_orig, cx + side // 2)
                    y2 = min(h_orig, cy + side // 2)

                    if x1 < x2 and y1 < y2:
                        crop = bgr_orig[y1:y2, x1:x2].copy()
                        rect_x1 = x - x1
                        rect_y1 = y - y1
                        rect_x2 = x + w - x1
                        rect_y2 = y + h - y1
                        
                        cv2.rectangle(crop, (rect_x1, rect_y1), (rect_x2, rect_y2), (0,255,0), 3)
                        
                        snippet_path = Config.SNIP / f"{img.id}_{typ}_{fid_counter}.jpg"
                        cv2.imwrite(str(snippet_path), crop)
                except Exception as e:
                    logging.warning(f"Failed to create snippet for {img.id}: {e}")

                risk = self._risk_score(typ, final_score, img.ts)
                mit = self._mitigation(typ)
                
                findings.append(Finding(
                    fid=fid_counter,
                    img_path=img.path,
                    typ=typ,
                    score=final_score,
                    desc=f"Cls: {conf:.2f}, DL: {dl_boost:.2f}",
                    lat=img.lat, lon=img.lon,
                    bbox=(x,y,w,h),
                    snippet_path=snippet_path,
                    classical_conf=conf, 
                    dl_boost=dl_boost,
                    risk_score=risk,
                    mitigation=mit,
                    ts=img.ts
                ))
                fid_counter += 1
                
        return findings

    def _risk_score(self, typ: str, conf: float, ts: Optional[datetime]) -> float:
        base = {
            "camera": 8.1, "entry": 7.3, "graffiti": 4.7, "person": 6.0, 
            "scene_change": 8.6,
            "license_plate": 9.0, "stop_sign": 8.3, "speed_sign": 7.7, "face": 8.8,
            "sign_obstructed": 9.5 # NEW: High risk for obstructed signs
        }
        
        now = datetime.now()
        age_days = (now - (ts or now)).days

        time_decay = max(0.5, 1.0 - (age_days / 365.0) * 0.1)
        
        return round(base.get(typ, 5.0) * conf * time_decay, 1)

    def _mitigation(self, typ: str) -> str:
        m = {
            "camera": "Encrypt feed, restrict access, use anti-tamper seals.",
            "entry": "Deploy RFID/biometric locks, audit logs.",
            "graffiti": "Anti-graffiti coating, increase patrols.",
            "license_plate": "Mask in footage, restrict access to LPR data.",
            "stop_sign": "Verify visibility, replace if damaged.",
            "speed_sign": "Verify visibility and local speed laws.",
            "scene_change": "Investigate immediately, correlate with logs.",
            "face": "Mask faces, comply with privacy laws.",
            "sign_obstructed": "Immediate replacement required. Sign is non-compliant and a safety hazard."
        }
        return m.get(typ, "Monitor and review.")

# ----------------------------------------------------------------------
# L36: Multiple Object Tracking (Scene Change Detection)
# ----------------------------------------------------------------------
class Tracker:
    def __init__(self):
        self.lk_params = dict(
            winSize=(15,15), 
            maxLevel=2, 
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    def detect_scene_changes(self, imgs: List[Img]) -> List[Finding]:
        logging.info("Running Scene Change Detection...")
        groups = collections.defaultdict(list)
        for im in imgs:
            if im.ts is None: continue 
            groups[(round(im.lat, 6), round(im.lon, 6))].append(im)
        
        findings = []
        fid = 100000 
        
        for loc, img_list in groups.items():
            if len(img_list) < 2: continue
            
            img_list.sort(key=lambda x: x.ts)
            
            try:
                prev_gray = cv2.imdecode(np.frombuffer(img_list[0].data, np.uint8), cv2.IMREAD_GRAYSCale)
                prev_gray = cv2.equalizeHist(prev_gray)
            except Exception:
                continue

            for img in img_list[1:]:
                try:
                    curr_gray = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_GRAYSCALE)
                    curr_gray = cv2.equalizeHist(curr_gray)
                except Exception:
                    prev_gray = None; break
                
                if prev_gray is None:
                    prev_gray = curr_gray
                    continue

                if prev_gray.shape != curr_gray.shape:
                    try:
                        h, w = prev_gray.shape[:2]
                        curr_gray = cv2.resize(curr_gray, (w, h), interpolation=cv2.INTER_AREA)
                    except Exception as e:
                        logging.warning(f"Failed to resize image for tracking: {e}")
                        prev_gray = curr_gray 
                        continue

                p0 = cv2.goodFeaturesToTrack(
                    prev_gray, 
                    maxCorners=Config.TRACK_MAX_CORNERS, 
                    qualityLevel=Config.TRACK_QUAL_LEVEL, 
                    minDistance=Config.TRACK_MIN_DIST
                )
                if p0 is None:
                    prev_gray = curr_gray
                    continue

                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
                
                if p1 is None or st is None:
                    prev_gray = curr_gray
                    continue
                    
                good_new = p1[st.ravel() == 1]
                good_old = p0[st.ravel() == 1]
                
                if len(good_new) > Config.TRACK_MIN_GOOD_FEAT:
                    displacement = good_new - good_old
                    avg_motion = np.linalg.norm(np.mean(displacement, axis=0))
                    
                    if avg_motion > Config.TRACK_MOTION_TH:
                        
                        age_days = (datetime.now() - (img.ts or datetime.now())).days
                        if age_days > 730:
                            continue # Skip this detection, it's too old
                        
                        risk = self._risk_score("scene_change", 0.97, img.ts)
                        
                        findings.append(Finding(
                            fid, img.path, "scene_change", 0.97,
                            f"Significant Scene Change (Motion: {avg_motion:.2f}px)",
                            loc[0], loc[1], (0,0,0,0), None,
                            0.97, 0.0, risk, 
                            self._mitigation("scene_change"),
                            ts=img.ts
                        ))
                        fid += 1
                
                prev_gray = curr_gray
                
        logging.info(f"Found {len(findings)} significant scene changes.")
        return findings

    def _risk_score(self, typ: str, conf: float, ts: Optional[datetime]) -> float:
        base = {"scene_change": 8.6}
        now = datetime.now()
        age_days = (now - (ts or now)).days
        
        time_decay = max(0.5, 1.0 - (age_days / 365.0) * 0.1)
        return round(base.get(typ, 5.0) * conf * time_decay, 1)

    def _mitigation(self, typ: str) -> str:
        return "Investigate immediately, correlate with logs."

# ----------------------------------------------------------------------
# PDF REPORT 
# ----------------------------------------------------------------------
from reportlab.graphics.shapes import Drawing, Line

class Report:
    def __init__(self, name: str, lat: float, lon: float, findings: List[Finding], imgs: List[Img]):
        self.pdf = Config.OUTPUT / f"REPORT_{name}.pdf"
        self.doc = SimpleDocTemplate(str(self.pdf), pagesize=letter, leftMargin=0.7*inch, rightMargin=0.7*inch)
        self.styles = getSampleStyleSheet()
        self.story = []
        self.name = name
        self.lat = lat
        self.lon = lon
        self.findings = sorted(findings, key=lambda x: x.risk_score, reverse=True)
        self.imgs = imgs

        # --- Custom Styles ---
        self.styles.add(ParagraphStyle(name='Header', fontSize=18, fontName='Helvetica-Bold'))
        self.styles.add(ParagraphStyle(name='SubHeader', fontSize=11, fontName='Helvetica', spaceAfter=12))
        self.styles.add(ParagraphStyle(name='Section', fontSize=14, fontName='Helvetica-Bold', spaceBefore=12, spaceAfter=6))
        self.styles.add(ParagraphStyle(name='Footer', fontSize=9, fontName='Helvetica-Oblique', alignment=TA_CENTER))
        
        # Style for table headers
        self.styles.add(ParagraphStyle(name='TableHeader', fontSize=10, fontName='Helvetica-Bold', textColor=colors.white, alignment=TA_CENTER, leading=12))
        # Style for text inside table cells
        self.styles.add(ParagraphStyle(name='TableCell', fontSize=9, fontName='Helvetica', leading=11))
        # Style for right-aligned table cell text
        self.styles.add(ParagraphStyle(name='TableCellRight', fontSize=9, fontName='Helvetica', alignment=TA_CENTER))

        self.styles.add(ParagraphStyle(
            name='TableCellIndented', 
            parent=self.styles['TableCell'], 
            leftIndent=0.25*inch
        ))

    def _build_header(self):
        self.story.append(Paragraph(f"OSINT THREAT REPORT", self.styles['Header']))
        self.story.append(Paragraph(f"<b>Target:</b> {self.name}", self.styles['SubHeader']))

        hr = Drawing(1, 1)
        hr.add(Line(0, 0, 500, 0, strokeColor=colors.black, strokeWidth=0.5))
        self.story.append(hr)
        self.story.append(Spacer(1, 0.2*inch))

    def _build_summary(self):
        self.story.append(Paragraph("<b>Executive Summary</b>", self.styles['Section']))
        
        total = len(self.findings)
        critical = sum(1 for f in self.findings if f.risk_score >= 7.0)
        high = sum(1 for f in self.findings if 5.0 <= f.risk_score < 7.0)
        by_type = collections.Counter(f.typ for f in self.findings)

        summary_data = [
            [Paragraph('<b>Location:</b>', self.styles['TableCell']), Paragraph(f"{self.lat:.6f}, {self.lon:.6f}", self.styles['TableCell'])],
            [Paragraph('<b>Total Detections:</b>', self.styles['TableCell']), Paragraph(f"{total}", self.styles['TableCell'])],
            [Paragraph('<b>Critical Risk (≥ 7.0):</b>', self.styles['TableCell']), Paragraph(f"{critical}", self.styles['TableCell'])],
            [Paragraph('<b>High Risk (5.0–6.9):</b>', self.styles['TableCell']), Paragraph(f"{high}", self.styles['TableCell'])],
            [Paragraph('<b>Report Generated:</b>', self.styles['TableCell']), Paragraph(f"{datetime.now():%Y-%m-%d %H:%M} UTC", self.styles['TableCell'])],
        ]
        
        summary_table = Table(summary_data, colWidths=['25%', '75%'])
        summary_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        
        self.story.append(summary_table)
        self.story.append(Spacer(1, 0.1*inch))

        if by_type:
            self.story.append(Paragraph(f"<b>Breakdown by Type:</b>", self.styles['TableCell']))
            for k,v in by_type.most_common():
                type_name = k.title().replace('_', ' ')
                self.story.append(Paragraph(f"• {type_name}: {v}", self.styles['TableCellIndented']))
        
        self.story.append(Spacer(1, 0.2*inch))

    def _build_map(self):
        m = folium.Map([self.lat, self.lon], zoom_start=18, tiles="CartoDB positron")
        folium.Marker(
            [self.lat, self.lon], 
            popup="Target Location", 
            icon=folium.Icon(color='black', icon='crosshairs')
        ).add_to(m)
        
        for f in self.findings:
            if f.typ == "scene_change": continue 
            color = 'red' if f.risk_score >= 7 else 'orange' if f.risk_score >= 5 else 'blue'
            folium.CircleMarker(
                [f.lat, f.lon], 
                radius=6, 
                color=color, 
                fill=True, 
                fill_color=color, 
                fill_opacity=0.7,
                popup=f"<b>{f.typ.title().replace('_', ' ')}</b><br/>Risk: {f.risk_score}<br/>ID: {f.fid}"
            ).add_to(m)
            
        map_path = Config.OUTPUT / f"threat_map_{self.name}.html"
        m.save(str(map_path))
        logging.info(f"Interactive map saved to {map_path}")
        self.story.append(Paragraph(f"<b>Interactive Threat Map:</b> See attached <u>{map_path.name}</u>", self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))

    def _build_findings_table(self):
        self.story.append(Paragraph("<b>Detailed Findings (Top 20)</b>", self.styles['Section']))
        
        if not self.findings:
            self.story.append(Paragraph("No high-confidence detections found.", self.styles['Normal']))
            return
            
        headers = [
            Paragraph("ID", self.styles['TableHeader']),
            Paragraph("Type", self.styles['TableHeader']),
            Paragraph("Risk", self.styles['TableHeader']),
            Paragraph("Score", self.styles['TableHeader']),
            Paragraph("Details", self.styles['TableHeader']),
            Paragraph("Location (Lat, Lon)", self.styles['TableHeader']),
            Paragraph("Snippet", self.styles['TableHeader'])
        ]
        data = [headers]
        
        col_widths = [
            0.4*inch,  # ID
            0.9*inch,  # Type
            0.5*inch,  # Risk
            0.5*inch,  # Score
            1.4*inch,  # Details
            1.7*inch,  # Location
            1.7*inch   # Snippet 
        ]
        
        for f in self.findings[:20]:
            img_cell = Paragraph("No Snippet", self.styles['TableCell']) # Default
            if f.snippet_path and f.snippet_path.exists():
                try:
                    img_cell = Image(str(f.snippet_path), width=1.5*inch, height=1.2*inch)
                    img_cell.hAlign = 'CENTER'
                    img_cell = KeepInFrame(1.5*inch, 1.2*inch, [img_cell], hAlign='center', vAlign='middle')
                except Exception as e:
                    logging.warning(f"Could not read snippet {f.snippet_path}: {e}")
                    img_cell = Paragraph("(Error Loading)", self.styles['TableCell'])
                    
            data.append([
                Paragraph(str(f.fid), self.styles['TableCellRight']),
                Paragraph(f.typ.title().replace('_', ' '), self.styles['TableCell']),
                Paragraph(f"{f.risk_score:.1f}", self.styles['TableCellRight']), 
                Paragraph(f"{f.score:.2f}", self.styles['TableCellRight']),
                Paragraph(f.desc, self.styles['TableCell']), 
                Paragraph(f"{f.lat:.5f},<br/>{f.lon:.5f}", self.styles['TableCell']),
                img_cell
            ])

        tbl = Table(data, colWidths=col_widths, rowHeights=1.3*inch) 
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E4057')),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BOX', (0,0), (-1,-1), 1, colors.black),
            ('ALIGN', (0,1), (5,-1), 'LEFT'),
            ('VALIGN', (0,1), (5,-1), 'TOP'),
            ('ALIGN', (2,1), (3,-1), 'CENTER'),
        ]))
        self.story.append(tbl)
        self.story.append(PageBreak())

    def _build_mitigation_matrix(self):
        self.story.append(Paragraph("<b>Mitigation Matrix</b>", self.styles['Section']))
        
        unique_threats = {}
        for f in self.findings:
            if f.typ not in unique_threats or f.risk_score > unique_threats[f.typ]['risk']:
                unique_threats[f.typ] = {'risk': f.risk_score, 'action': f.mitigation}
        
        if not unique_threats:
            self.story.append(Paragraph("No mitigations required.", self.styles['Normal']))
            return

        headers = [
            Paragraph("Threat", self.styles['TableHeader']),
            Paragraph("Max Risk", self.styles['TableHeader']),
            Paragraph("Recommended Action", self.styles['TableHeader'])
        ]
        mit_data = [headers]
        
        sorted_threats = sorted(unique_threats.items(), key=lambda item: item[1]['risk'], reverse=True)
        
        for typ, data in sorted_threats:
            mit_data.append([
                Paragraph(typ.title().replace('_', ' '), self.styles['TableCell']), 
                Paragraph(f"{data['risk']:.1f}", self.styles['TableCellRight']), 
                Paragraph(data['action'], self.styles['TableCell'])
            ])
            
        mit_tbl = Table(mit_data, colWidths=['25%', '15%', '60%']) 
        mit_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E4057')), 
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.black),
            ('BOX', (0,0), (-1,-1), 1, colors.black),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('ALIGN', (1,1), (1,-1), 'CENTER'), 
        ]))
        self.story.append(mit_tbl)

    def _build_timeline(self):
        self.story.append(PageBreak())
        self.story.append(Paragraph("<b>Detection Timeline (Summarized)</b>", self.styles['Section']))
        
        timeline_findings = [f for f in self.findings if f.ts]
        if not timeline_findings:
            self.story.append(Paragraph("No timestamped detections found.", self.styles['Normal']))
            return
            
        timeline = sorted(timeline_findings, key=lambda x: x.ts)
        
        # Group by Date
        grouped_by_date = collections.defaultdict(list)
        for f in timeline:
            grouped_by_date[f.ts.strftime("%Y-%m-%d")].append(f)

        for date_str, daily_findings in sorted(grouped_by_date.items()):
            self.story.append(Spacer(1, 0.1*inch))
            self.story.append(Paragraph(f"<b>{date_str}</b>", self.styles['Normal']))
            
            # Group by Hour
            grouped_by_hour = collections.defaultdict(list)
            for f in daily_findings:
                grouped_by_hour[f.ts.strftime("%H")].append(f)
                
            for hour_str, hourly_findings in sorted(grouped_by_hour.items()):
                count = len(hourly_findings)
                if count == 0: continue
                
                # Get type counts
                types_count = collections.Counter(f.typ.title().replace('_', ' ') for f in hourly_findings)
                types_str = ", ".join(f"{k} ({v})" for k,v in types_count.most_common(3))
                if len(types_count) > 3:
                    types_str += ", ..."
                
                # Get average location
                avg_lat = sum(f.lat for f in hourly_findings) / count
                avg_lon = sum(f.lon for f in hourly_findings) / count
                
                text = (
                    f"• <font face='Courier'>{hour_str}:00:xx</font> – <b>{count} detection{'s' if count > 1 else ''}</b> "
                    f"({types_str}) clustered at <i>{avg_lat:.4f}, {avg_lon:.4f}</i>"
                )
                self.story.append(Paragraph(text, self.styles['TableCellIndented']))


    def _build_footer(self, canvas, doc):
        canvas.saveState()
        footer = Paragraph("Generated by OSINT-VTM v23.5-WORLD (CSE3172)", self.styles['Footer'])
        w, h = footer.wrap(doc.width, doc.bottomMargin)
        footer.drawOn(canvas, doc.leftMargin, h)
        
        page_num = f"Page {doc.page}"
        canvas.setFont('Helvetica-Oblique', 9)
        canvas.drawRightString(doc.width + doc.leftMargin, h, page_num)
        canvas.restoreState()

    def generate(self):
        """Builds all parts and saves the PDF."""
        try:
            self._build_header()
            self._build_summary()
            self._build_map()
            self._build_findings_table()
            self.story.append(Paragraph("<i>*Risk Score: A time-decayed value based on detection type and confidence.</i>", self.styles['Footer']))
            self._build_mitigation_matrix()
            self._build_timeline()
            
            self.doc.build(self.story, onFirstPage=self._build_footer, onLaterPages=self._build_footer)
        except Exception as e:
            logging.error(f"Failed to build PDF report: {e}")
            import traceback
            traceback.print_exc()

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def setup_logging():
    """Configures logging."""
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def detection_worker(img: Img) -> List[Finding]:
    """
    Initializes all models *inside* the worker process to avoid pickling.
    """
    global YOLO_MODEL, CLIP_MODEL, CLIP_PREPROCESS, DEVICE, Config
    
    # 1. Load YOLOWorld (from local file)
    try:
        from ultralytics import YOLO 
        yolo_path = Config.MODELS / "yolov8x-worldv2.pt" # <-- Using the 'x' v2 model
        
        if not yolo_path.exists():
            raise FileNotFoundError(f"Worker could not find {yolo_path}")
            
        YOLO_MODEL = YOLO(str(yolo_path)) # Load from local path
        YOLO_MODEL.model.eval()
    except Exception as e:
        logging.error(f"Worker failed to load YOLO: {e}")
        YOLO_MODEL = None
    
    # 2. Load CLIP (using the correct library)
    try:
        import torch, clip
        DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
        CLIP_MODEL, CLIP_PREPROCESS = clip.load('ViT-B/32', device=DEVICE)
        CLIP_MODEL.eval()
    except Exception as e:
        logging.error(f"Worker failed to load CLIP: {e}")
        CLIP_MODEL = None

    detector = V23Detector()
    return detector.detect(img)

def filter_findings(findings: List[Finding], strict: bool) -> List[Finding]:
    """Filters a list of raw findings based on strict or lenient thresholds."""
    if strict:
        th_ver = Config.CLASSICAL_TH_VERIFIED
        th_unver = Config.CLASSICAL_TH_UNVERIFIED
        logging.info(f"Applying STRICT filter (Verified: {th_ver}, Unverified: {th_unver})")
    else:
        th_ver = Config.LENIENT_TH_VERIFIED
        th_unver = Config.LENIENT_TH_UNVERIFIED
        logging.info(f"Applying LENIENT filter (Verified: {th_ver}, Unverified: {th_unver})")
        
    filtered = []
    for f in findings:
        if f.dl_boost > 0.0:
            if f.classical_conf >= th_ver:
                filtered.append(f)
        else:
            if f.classical_conf >= th_unver:
                filtered.append(f)
    return filtered

def process_obstructions(findings: List[Finding]) -> List[Finding]:
    """Checks for graffiti on signs and re-classifies them."""
    logging.info("Checking for sign obstructions...")
    sign_types = {"stop_sign", "speed_sign"}
    signs = [f for f in findings if f.typ in sign_types]
    graffiti = [f for f in findings if f.typ == "graffiti"]
    
    if not signs or not graffiti:
        logging.info("No signs or graffiti found to compare.")
        return findings

    detector = V23Detector() # For _iou and _risk_score methods
    obstructed_count = 0
    
    for sign in signs:
        sign_box = (sign.bbox[0], sign.bbox[1], sign.bbox[0] + sign.bbox[2], sign.bbox[1] + sign.bbox[3])
        
        for graf in graffiti:
            # Only check if from the same image
            if sign.img_path != graf.img_path:
                continue
                
            graf_box = (graf.bbox[0], graf.bbox[1], graf.bbox[0] + graf.bbox[2], graf.bbox[1] + graf.bbox[3])
            
            # Use IoU to check for overlap
            iou = detector._iou(sign_box, graf_box)
            
            if iou > 0.1: # 10% overlap is enough to obstruct
                logging.info(f"Found obstructed sign on {sign.img_path.name}")
                sign.typ = "sign_obstructed"
                sign.mitigation = detector._mitigation(sign.typ)
                # Recalculate risk score with new, higher-risk type
                sign.risk_score = detector._risk_score(sign.typ, sign.score, sign.ts)
                obstructed_count += 1
                break # Move to the next sign
                
    logging.info(f"Re-classified {obstructed_count} signs as 'sign_obstructed'.")
    return findings

# --- main() FUNCTION ---
def main():
    p = argparse.ArgumentParser(description="OSINT-VTM v23.5-WORLD - Visual Threat Mapper")
    p.add_argument("--lat", type=float, required=True, help="Target latitude (e.g., 40.7128)")
    p.add_argument("--lon", type=float, required=True, help="Target longitude (e.g., -74.0060)")
    p.add_argument("--radius", type=int, default=150, help="Scan radius in meters (default: 150)")
    p.add_argument("--name", type=str, default="TARGET", help="Name for the report file (default: TARGET)")
    args = p.parse_args()

    setup_logging()
    
    # --- 1. Init & Download ---
    logging.info("OSINT-VTM v23.5-WORLD – INITIALIZING (UPGRADED-PRECISION)")
    Config.init() # Downloads models if needed
    
    # --- 2. Fetch Data ---
    imgs = fetch_images(args.lat, args.lon, args.radius)
    if not imgs:
        logging.error("No recent images found for the specified location. Exiting.")
        return

    tracker = Tracker()
    all_raw_findings = [] 

    # --- 3. Run Detection Pipeline ---
    with ProcessPoolExecutor(max_workers=Config.WORKERS, initializer=Config.init) as pool:
        futures = [pool.submit(detection_worker, im) for im in imgs]
        
        for f in tqdm(as_completed(futures), total=len(futures), desc="Detect (Cls+DL)"):
            try:
                all_raw_findings.extend(f.result()) 
            except Exception as e:
                logging.error(f"Error in detection worker: {e}")
                
    logging.info(f"Collected {len(all_raw_findings)} raw (unfiltered) detections.")

    # --- 4. Dynamic Filtering ---
    all_findings = filter_findings(all_raw_findings, strict=True)
    
    if not all_findings and all_raw_findings: 
        logging.warning("No detections passed strict filter. Re-running with lenient filter.")
        all_findings = filter_findings(all_raw_findings, strict=False)
    
    logging.info(f"Found {len(all_findings)} final high-confidence detections.")

    # --- 5. Run Change Detection ---
    all_findings.extend(tracker.detect_scene_changes(imgs))

    # --- 6. NEW: Run Obstruction Post-Processing ---
    all_findings = process_obstructions(all_findings)

    # --- 7. Generate Report ---
    if not all_findings:
        logging.warning("No findings discovered. Report will be minimal.")
    
    logging.info(f"Generating PDF report for {args.name}...")
    report = Report(args.name, args.lat, args.lon, all_findings, imgs)
    report.generate()
    
    logging.info(f"--- ANALYSIS COMPLETE ---")
    logging.info(f"Report  → {report.pdf.resolve()}")
    logging.info(f"Map     → {(Config.OUTPUT / f'threat_map_{args.name}.html').resolve()}")
    logging.info(f"Snippets → {Config.SNIP.resolve()}")

if __name__ == "__main__":
    if not MAPILLARY_TOKEN:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!! ERROR: MAPILLARY_ACCESS_TOKEN is not set.              !!")
        print("!! Please edit the script (around line 256) to set your token. !!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        main()