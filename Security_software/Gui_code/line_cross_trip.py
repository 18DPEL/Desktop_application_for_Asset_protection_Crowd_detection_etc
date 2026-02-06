import cv2
import time
import torch
import numpy as np
import threading
from datetime import datetime
from ultralytics import YOLO
import requests
import base64
import json
import os
from collections import defaultdict, deque

# Import robust RTSP reader
import sys
sys.path.insert(0, r"D:\LLM_SETUP\Scurity_software\RTSP_Reader")
try:
    from rtsp_reader import RobustRTSPReader, RTSPConfig, DecodingMode
except ImportError:
    print("Warning: RobustRTSPReader not found. Please ensure the library is in the correct path.")
    # Fallback or exit? For now, we assume it exists as per instructions.
    pass

# -------------------------------------------------
# HARD CONFIG
# -------------------------------------------------
MODEL_PATH = "yolov8n.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FRAME_WIDTH = 960
FRAME_HEIGHT = 480
QUEUE_SIZE = 3
MICRO_BATCH_SIZE = 5

# Tracking Parameters
TRACKING_HISTORY_SIZE = 30  # Number of history points to keep for intersection check

# API Configuration (Using same base IP as provided examples)
API_ENDPOINT = "http://192.168.1.7/cameradetection/public/api/line_cross"  # Hypothethical endpoint
API_TIMEOUT = 5

# Detection classes 
DETECTION_CLASSES = {
    0: 'person',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    # Add other classes as needed
}

CONFIDENCE = 0.4

# Config file for saving lines
TRIPWIRE_CONFIG_FILE = r"D:\clean_puri_hall_project\Crowd_Detection\tripwire_config.json"

# -------------------------------------------------
# MATH UTILS: LINE INTERSECTION
# -------------------------------------------------
def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A, B, C, D):
    """
    Return true if line segments AB and CD intersect
    """
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# -------------------------------------------------
# MICRO BATCH INFERENCER
# -------------------------------------------------
class MicroBatchInferencer:
    """Batch inference for improved FPS on GPU"""
    def __init__(self, model, batch_size=MICRO_BATCH_SIZE):
        self.model = model
        self.batch_size = batch_size
    
    def infer(self, frames):
        """Run batch inference on multiple frames"""
        results = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            with torch.inference_mode():
                preds = self.model(
                    batch,
                    conf=CONFIDENCE,
                    classes=list(DETECTION_CLASSES.keys()),
                    device=DEVICE,
                    verbose=False
                )
            results.extend(preds)
        return results
    
    def track(self, frame, persist=True):
        """Run tracking on a single frame"""
        with torch.inference_mode():
            result = self.model.track(
                frame, 
                persist=persist, 
                verbose=False,
                conf=CONFIDENCE,
                iou=0.4,
                classes=list(DETECTION_CLASSES.keys()),
                imgsz=640,
                device=DEVICE
            )
        return result

# -------------------------------------------------
# ASYNC API SENDER
# -------------------------------------------------
def send_to_server(payload, endpoint=API_ENDPOINT):
    try:
        requests.post(endpoint, json=payload, timeout=API_TIMEOUT)
    except Exception as e:
        print(f"[API ERROR] {e}", flush=True)

# -------------------------------------------------
# LOGGING + API HELPER
# -------------------------------------------------
def log_crossing_alert(cam_id, area_name, line_name, obj_data, direction):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    track_id = obj_data['track_id']
    object_type = obj_data['object_type']
    
    print(f"[{timestamp}] 🚨 CAM-{cam_id + 1} | Area={area_name} | "
          f"LINE CROSSED ({line_name}): {object_type} (ID:{track_id}) | Dir: {direction}", flush=True)
    
    payload = {
        "timestamp": timestamp,
        "camera_id": cam_id + 1,
        "area": area_name,
        "alert_type": "LINE_CROSS",
        "line_name": line_name,
        "track_id": track_id,
        "object_type": object_type,
        "object_class": obj_data.get('class_id', 0),
        "confidence": obj_data.get('confidence', 0),
        "direction": direction,
        "message": f"Object {object_type} crossed line {line_name}"
    }
    
    # API sending disabled
    # threading.Thread(target=send_to_server, args=(payload,), daemon=True).start()

# -------------------------------------------------
# TRIPWIRE DETECTOR CLASS
# -------------------------------------------------
class LineTripwireDetector:
    def __init__(self, model_path=MODEL_PATH, confidence=CONFIDENCE, device=DEVICE,
                 batch_size=MICRO_BATCH_SIZE):
        self.model = YOLO(model_path).to(device)
        self.confidence = confidence
        self.device = device
        
        # Batch inferencer
        self.batch_inferencer = MicroBatchInferencer(self.model, batch_size)
        
        # Tripwire State
        # tripwires structure: { 'line_0': {'p1': (x,y), 'p2': (x,y), 'count': 0}, ... }
        self.tripwires = {} 
        self.detection_active = False
        
        # Tracking history
        self.object_history = defaultdict(lambda: deque(maxlen=TRACKING_HISTORY_SIZE))
        
        self.frame_count = 0
        self.load_config()

    def load_config(self):
        if os.path.exists(TRIPWIRE_CONFIG_FILE):
            try:
                with open(TRIPWIRE_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    # Convert lists back to tuples if necessary
                    self.tripwires = data.get('tripwires', {})
                    # Ensure points are tuples
                    for name, data in self.tripwires.items():
                        data['p1'] = tuple(data['p1'])
                        data['p2'] = tuple(data['p2'])
                print(f"✓ Loaded {len(self.tripwires)} tripwires")
            except Exception as e:
                print(f"Error loading tripwires: {e}")
                self.tripwires = {}

    def save_config(self):
        try:
            data = {'tripwires': self.tripwires}
            with open(TRIPWIRE_CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved {len(self.tripwires)} tripwires")
        except Exception as e:
            print(f"Error saving tripwires: {e}")

    def add_tripwire(self, p1, p2):
        name = f"line_{len(self.tripwires) + 1}"
        self.tripwires[name] = {
            'p1': p1,
            'p2': p2,
            'count': 0,
            'color': (0, 255, 255) # Default Yellow
        }
        print(f"Added tripwire: {name} from {p1} to {p2}")

    def clear_tripwires(self):
        self.tripwires = {}
        print("Cleared all tripwires")
        
    def reset_polygon(self):
        """Alias for compatibility with GUI"""
        self.clear_tripwires()
        self.detection_active = False

    def start_detection(self):
        self.detection_active = True
        print("▶️ Detection STARTED")

    def stop_detection(self):
        self.detection_active = False
        print("⏹️ Detection STOPPED")

    def process(self, frame, cam_id=0, area_name="default"):
        self.frame_count += 1
        
        alerts = []
        detected_objects = []
        
        # 1. Tracking
        results = self.batch_inferencer.track(frame, persist=True)
        
        current_track_ids = set()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            has_ids = boxes.id is not None
            
            for i, box in enumerate(boxes):
                if not has_ids: continue
                
                track_id = int(boxes.id[i])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Bbox info
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                object_type = DETECTION_CLASSES.get(cls_id, f"class_{cls_id}")
                current_track_ids.add(track_id)
                
                # Update history
                prev_center = None
                if len(self.object_history[track_id]) > 0:
                    prev_center = self.object_history[track_id][-1]
                    
                self.object_history[track_id].append(center)
                
                obj_data = {
                    'track_id': track_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'object_type': object_type,
                    'confidence': conf
                }
                detected_objects.append(obj_data)
                
                # 2. Crossing Logix
                if self.detection_active and prev_center is not None:
                    # Check intersection with ALL tripwires
                    for name, line in self.tripwires.items():
                        L1 = line['p1']
                        L2 = line['p2']
                        
                        # Only check if the movement is significant to avoid noise
                        dist = ((center[0]-prev_center[0])**2 + (center[1]-prev_center[1])**2)**0.5
                        if dist < 2: continue

                        if intersect(prev_center, center, L1, L2):
                            # Crossed!
                            line['count'] += 1
                            
                            # Trigger Alert
                            # Determine direction
                            log_crossing_alert(cam_id, area_name, name, obj_data, "Crossed")
                            
                            alerts.append({
                                'line': name,
                                'obj': obj_data
                            })
                            
                            # Visual feedback timer
                            line['last_cross_time'] = time.time()

        # Clean up stale history
        if self.frame_count % 30 == 0:
            active_ids = current_track_ids
            # Optional cleanup

        return {'detected_objects': detected_objects, 'alerts': alerts, 'tripwires': self.tripwires}

    def draw(self, frame, result):
        overlay = frame.copy()
        
        # 1. Draw Tripwires
        for name, line in self.tripwires.items():
            p1 = line['p1']
            p2 = line['p2']
            
            # Color logic: Flash RED if recently crossed
            color = line.get('color', (0, 255, 255))
            if time.time() - line.get('last_cross_time', 0) < 1.0:
                 color = (0, 0, 255) # Red flash
            
            cv2.line(frame, p1, p2, color, 3)
            
            # Draw Endpoints
            cv2.circle(frame, p1, 5, (0, 200, 255), -1)
            cv2.circle(frame, p2, 5, (0, 200, 255), -1)
            
            # Draw Name & Count
            mid_x = (p1[0] + p2[0]) // 2
            mid_y = (p1[1] + p2[1]) // 2
            
            text = f"{name}: {line['count']}"
            cv2.putText(frame, text, (mid_x, mid_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(frame, (mid_x-5, mid_y-25), (mid_x+100, mid_y-5), (0,0,0), -1) # bg
            cv2.putText(frame, text, (mid_x, mid_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # 2. Draw Objects
        for obj in result.get('objects', []):
            x1, y1, x2, y2 = obj['bbox']
            label = f"{obj['object_type']} {obj['track_id']}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

# -------------------------------------------------
# ARGUMENT PARSING
# -------------------------------------------------
import argparse
def parse_args():
    parser = argparse.ArgumentParser("Line Crossing Tripwire System")
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--camera", action="append", required=True)
    parser.add_argument("--area", action="append", required=True)
    args = parser.parse_args()

    if len(args.camera) != len(args.area):
        raise ValueError("Each --camera must have a matching --area")
    if args.mode == "single" and len(args.camera) != 1:
        raise ValueError("Single mode requires exactly one camera")
    return args

# -------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------
def run_line_tripwire():
    args = parse_args()
    RTSP_URLS = args.camera
    AREA_NAMES = args.area

    # ---------------- DEVICE ----------------
    print(f"\n>>> USING DEVICE: {DEVICE.upper()} <<<")
    
    # ---------------- CREATE DETECTORS ----------------
    detectors = [
        LineTripwireDetector() for _ in range(len(RTSP_URLS))
    ]
    
    # ---------------- CREATE STREAMS ----------------
    rtsp_configs = [
        RTSPConfig(
            decoding_mode=DecodingMode.AUTO,
            buffer_size=QUEUE_SIZE,
            infinite_reconnect=True
        ) for _ in RTSP_URLS
    ]
    
    streams = [
        RobustRTSPReader(url, config=cfg).start()
        for url, cfg in zip(RTSP_URLS, rtsp_configs)
    ]
    
    # ---------------- MOUSE STATE ----------------
    # State: 0=None, 1=FirstPointClicked
    mouse_state = [{
        'state': 0,
        'p1': None,
        'current_pos': None
    } for _ in range(len(RTSP_URLS))]
    
    def mouse_callback(event, x, y, flags, cam_id):
        detector = detectors[cam_id]
        m_state = mouse_state[cam_id]
        
        m_state['current_pos'] = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if m_state['state'] == 0:
                # Start new line
                m_state['state'] = 1
                m_state['p1'] = (x, y)
                print(f"Start point set at {x},{y}")
            elif m_state['state'] == 1:
                # End line
                detector.add_tripwire(m_state['p1'], (x, y))
                m_state['state'] = 0
                m_state['p1'] = None
                print(f"End point set at {x},{y}. Line created.")
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Cancel drawing
            m_state['state'] = 0
            m_state['p1'] = None
            print("Cancelled drawing")

    # ---------------- WINDOWS ----------------
    for i in range(len(RTSP_URLS)):
        cv2.namedWindow(AREA_NAMES[i])
        cv2.setMouseCallback(AREA_NAMES[i], mouse_callback, i)
        
    prev_time = time.time()
    
    try:
        while True:
            frames = [s.get_frame() for s in streams]
            if any(f is None for f in frames):
                 time.sleep(0.01)
                 continue
                 
            now = time.time()
            fps = 1.0 / max(now - prev_time, 0.0001)
            prev_time = now
            
            for i, (frame, detector) in enumerate(zip(frames, detectors)):
                display_frame = frame.copy()
                
                # Logic
                result = detector.process(frame, i, AREA_NAMES[i])
                
                # Draw
                display_frame = detector.draw(display_frame, result)
                
                # Draw creation line preview
                m_state = mouse_state[i]
                if m_state['state'] == 1 and m_state['p1'] and m_state['current_pos']:
                     cv2.line(display_frame, m_state['p1'], m_state['current_pos'], (200, 200, 200), 2)
                
                # HUD
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                status_text = "MONITORING" if detector.detection_active else "IDLE (Press ENTER)"
                cv2.putText(display_frame, f"Status: {status_text}", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if detector.detection_active else (0, 0, 255), 2)
                
                instr = "L-Click: Draw Line | R-Click: Cancel | ENTER: Toggle | R: Reset | S: Save | ESC: Quit"
                cv2.putText(display_frame, instr, (20, FRAME_HEIGHT - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                cv2.imshow(AREA_NAMES[i], display_frame)
                
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 13: # Enter
                for d in detectors: d.toggle_detection()
            elif key == ord('r') or key == ord('R'):
                for d in detectors: d.clear_tripwires()
            elif key == ord('s') or key == ord('S'):
                for d in detectors: d.save_config()

    except KeyboardInterrupt:
        pass
    finally:
        for d in detectors: d.save_config()
        for s in streams: s.stop()
        cv2.destroyAllWindows()
        print("Closed.")

if __name__ == "__main__":
    run_line_tripwire()