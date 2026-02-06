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
from rtsp_reader import RobustRTSPReader, RTSPConfig, DecodingMode

# -------------------------------------------------
# HARD CONFIG
# -------------------------------------------------
MODEL_PATH = "yolov8n.pt"
DEVICE = "cuda"

FRAME_WIDTH = 960
FRAME_HEIGHT = 480
QUEUE_SIZE = 3
MICRO_BATCH_SIZE = 5

# Tracking Parameters
TRACKING_HISTORY_SIZE = 60  # frames to keep in tracking history
ALERT_INTERVAL = 10  # seconds between alerts for same object

# API Configuration
ASSET_API_ENDPOINT = "http://192.168.1.7/cameradetection/public/api/asset_protection"
API_TIMEOUT = 5

# Detection classes (objects to track)
DETECTION_CLASSES = {
    0: 'person',
    32: 'sports ball',
    29: 'frisbee',
    39: 'bottle', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    63: 'laptop', 64: 'mouse', 66: 'keyboard', 67: 'cell phone',
    73: 'book', 74: 'clock', 75: 'vase', 26: 'handbag', 28: 'suitcase',
    24: 'backpack', 25: 'umbrella', 27: 'tie'
}

# Detection confidence
CONFIDENCE = 0.4

# Polygon config file
POLYGON_CONFIG_FILE = r"D:\LLM_SETUP\Scurity_software\polygon_config.json"

# -------------------------------------------------
# MICRO BATCH INFERENCER FOR FASTER FPS
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
        """Run tracking on a single frame (for compatibility)"""
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
def send_to_server(payload, endpoint=ASSET_API_ENDPOINT):
    try:
        requests.post(endpoint, json=payload, timeout=API_TIMEOUT)
    except Exception as e:
        print(f"[API ERROR] {e}", flush=True)

# -------------------------------------------------
# LOGGING + API HELPER
# -------------------------------------------------
def log_zone_alert(cam_id, area_name, alert_info, frame=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    alert_type = alert_info['alert_type']
    object_type = alert_info['object_type']
    track_id = alert_info['track_id']
    
    if alert_type == "ASSET_LEFT_ZONE":
        message = f"🚨 ALERT: {object_type} (ID:{track_id}) has LEFT the protection zone!"
        print(f"[{timestamp}] 🚨 CAM-{cam_id + 1} | Area={area_name} | "
              f"ASSET NOT PROTECTED: {object_type} | Track ID: {track_id}", flush=True)
    elif alert_type == "ASSET_RETURNED":
        message = f"✓ {object_type} (ID:{track_id}) returned to protection zone"
        print(f"[{timestamp}] ✓ CAM-{cam_id + 1} | Area={area_name} | "
              f"ASSET RETURNED: {object_type}", flush=True)
    else:
        message = f"Unknown alert: {alert_type}"
    
    payload = {
        "timestamp": timestamp,
        "camera_id": cam_id + 1,
        "area": area_name,
        "alert_type": alert_type,
        "track_id": track_id,
        "object_type": object_type,
        "object_class": alert_info.get('class_id', 0),
        "confidence": alert_info.get('confidence', 0),
        "initial_position": alert_info.get('initial_position', [0, 0]),
        "current_position": alert_info.get('current_position', [0, 0]),
        "movement_path": alert_info.get('movement_path', []),
        "status": alert_info.get('status', 'unknown'),
        "message": message
    }
    
    # Encode frame as base64 if provided
    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        payload["frame"] = frame_b64
    
    # API sending disabled
    # threading.Thread(target=send_to_server, args=(payload,), daemon=True).start()

# -------------------------------------------------
# DRAW MOVEMENT PATH
# -------------------------------------------------
def draw_movement_path(frame, positions, color=(0, 255, 255), show_all=True):
    """Draw the movement path/trail for a tracked object"""
    if len(positions) < 2:
        return
    
    # Draw path lines with fading effect
    for i in range(1, len(positions)):
        pt1 = positions[i-1]
        pt2 = positions[i]
        
        if pt1 is not None and pt2 is not None:
            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))
            
            # Fade color based on age (older = more transparent)
            alpha = i / len(positions)
            line_color = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
            thickness = max(1, int(3 * (0.5 + 0.5 * alpha)))
            
            cv2.line(frame, pt1, pt2, line_color, thickness)
    
    # Draw dots at positions
    for i, pos in enumerate(positions):
        if i % 3 == 0 and pos is not None:  # Every 3rd point
            pt = tuple(map(int, pos))
            alpha = i / len(positions)
            point_color = tuple(int(c * (0.4 + 0.6 * alpha)) for c in color)
            cv2.circle(frame, pt, 3, point_color, -1)
    
    # Draw start and end points
    if positions:
        start_pt = tuple(map(int, positions[0]))
        cv2.circle(frame, start_pt, 6, (0, 255, 0), -1)  # Green start
        cv2.putText(frame, "S", (start_pt[0] - 5, start_pt[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        end_pt = tuple(map(int, positions[-1]))
        cv2.circle(frame, end_pt, 6, (0, 0, 255), -1)  # Red end

# -------------------------------------------------
# POLYGON ASSET PROTECTOR CLASS
# -------------------------------------------------
class PolygonAssetProtector:
    def __init__(self, model_path=MODEL_PATH, confidence=CONFIDENCE, device=DEVICE,
                 use_batch_inference=True, batch_size=MICRO_BATCH_SIZE):
        self.model = YOLO(model_path).to(device)
        self.confidence = confidence
        self.device = device
        
        # Micro batch inferencer for faster FPS
        self.use_batch_inference = use_batch_inference
        self.batch_inferencer = MicroBatchInferencer(self.model, batch_size)
        self.batch_size = batch_size
        
        # Polygon region (list of points)
        self.protection_polygon = []
        self.polygon_complete = False
        self.detection_active = False
        
        # Protected objects tracking
        self.protected_objects = {}  # track_id: object_data
        self.object_history = defaultdict(lambda: deque(maxlen=TRACKING_HISTORY_SIZE))
        
        self.frame_count = 0
        self.current_time = time.time()
        
        # Load saved polygon if exists
        self.load_polygon()
    
    def load_polygon(self):
        """Load saved polygon from configuration file"""
        if os.path.exists(POLYGON_CONFIG_FILE):
            try:
                with open(POLYGON_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                
                self.protection_polygon = [tuple(p) for p in data.get('polygon', [])]
                self.polygon_complete = len(self.protection_polygon) >= 3
                
                if self.polygon_complete:
                    print(f"✓ Loaded polygon with {len(self.protection_polygon)} points")
            except Exception as e:
                print(f"Error loading polygon: {e}")
                self.protection_polygon = []
    
    def save_polygon(self):
        """Save polygon to configuration file"""
        try:
            data = {
                'polygon': [list(p) for p in self.protection_polygon]
            }
            with open(POLYGON_CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved polygon with {len(self.protection_polygon)} points")
        except Exception as e:
            print(f"Error saving polygon: {e}")
    
    def add_polygon_point(self, point):
        """Add a point to the polygon"""
        self.protection_polygon.append(point)
        print(f"📍 Added point {len(self.protection_polygon)}: {point}")
    
    def complete_polygon(self):
        """Complete the polygon (close the shape)"""
        if len(self.protection_polygon) >= 3:
            self.polygon_complete = True
            self.save_polygon()
            print(f"✅ Polygon completed with {len(self.protection_polygon)} points!")
            return True
        else:
            print("⚠️ Need at least 3 points to complete polygon")
            return False
    
    def reset_polygon(self):
        """Reset the polygon and detection state"""
        self.protection_polygon = []
        self.polygon_complete = False
        self.detection_active = False
        self.protected_objects = {}
        self.object_history.clear()
        print("🔄 Polygon and detection reset")
    
    def start_detection(self):
        """Start detection after polygon is defined"""
        if not self.polygon_complete:
            print("⚠️ Cannot start detection - polygon not complete!")
            return False
        
        self.detection_active = True
        self.protected_objects = {}  # Reset protected objects
        self.object_history.clear()
        print("▶️ Detection started! Monitoring objects in protection zone...")
        return True
    
    def stop_detection(self):
        """Stop detection"""
        self.detection_active = False
        print("⏹️ Detection stopped")
    
    def is_inside_polygon(self, point):
        """Check if a point is inside the protection polygon"""
        if len(self.protection_polygon) < 3:
            return False
        
        polygon_np = np.array(self.protection_polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(polygon_np, (float(point[0]), float(point[1])), False)
        return result >= 0
    
    def process(self, frame):
        """Process a single frame for object detection and tracking"""
        self.current_time = time.time()
        self.frame_count += 1
        
        alerts = []
        detected_objects = []
        
        if not self.detection_active:
            return {
                'detected_objects': [],
                'protected_objects': self.protected_objects,
                'alerts': [],
                'frame_count': self.frame_count
            }
        
        # YOLO detection with tracking (using batch inferencer for speed)
        results = self.batch_inferencer.track(frame, persist=True)
        
        current_track_ids = set()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            has_ids = boxes.id is not None
            
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                track_id = int(boxes.id[i]) if has_ids else i
                conf = float(box.conf[0])
                
                object_type = DETECTION_CLASSES.get(cls_id, f"class_{cls_id}")
                current_track_ids.add(track_id)
                
                # Check if object is inside polygon
                is_inside = self.is_inside_polygon(center)
                
                # Store in object history
                self.object_history[track_id].append({
                    'time': self.current_time,
                    'position': center,
                    'inside': is_inside
                })
                
                obj_data = {
                    'track_id': track_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'class_id': cls_id,
                    'object_type': object_type,
                    'confidence': conf,
                    'is_inside': is_inside
                }
                detected_objects.append(obj_data)
                
                # Handle protected object tracking
                if track_id in self.protected_objects:
                    # Update existing protected object
                    pobj = self.protected_objects[track_id]
                    pobj['current_position'] = center
                    pobj['is_inside'] = is_inside
                    pobj['last_seen'] = self.current_time
                    pobj['bbox'] = (x1, y1, x2, y2)
                    
                    # Build movement path
                    movement_path = [h['position'] for h in self.object_history[track_id]]
                    pobj['movement_path'] = movement_path
                    
                    # Check if object has left the zone
                    if pobj['status'] == 'protected' and not is_inside:
                        # Object LEFT the protection zone!
                        pobj['status'] = 'left_zone'
                        pobj['left_at'] = self.current_time
                        
                        if self.current_time - pobj.get('last_alert_time', 0) > ALERT_INTERVAL:
                            alerts.append({
                                'alert_type': 'ASSET_LEFT_ZONE',
                                'track_id': track_id,
                                'object_type': object_type,
                                'class_id': cls_id,
                                'confidence': conf,
                                'initial_position': pobj['initial_position'],
                                'current_position': center,
                                'movement_path': movement_path[-30:],  # Last 30 positions
                                'status': 'left_zone'
                            })
                            pobj['last_alert_time'] = self.current_time
                    
                    # Check if object returned to zone
                    elif pobj['status'] == 'left_zone' and is_inside:
                        pobj['status'] = 'protected'
                        alerts.append({
                            'alert_type': 'ASSET_RETURNED',
                            'track_id': track_id,
                            'object_type': object_type,
                            'class_id': cls_id,
                            'confidence': conf,
                            'initial_position': pobj['initial_position'],
                            'current_position': center,
                            'movement_path': [],
                            'status': 'protected'
                        })
                
                else:
                    # New object - if inside zone, start protecting it
                    if is_inside:
                        self.protected_objects[track_id] = {
                            'track_id': track_id,
                            'object_type': object_type,
                            'class_id': cls_id,
                            'initial_position': center,
                            'current_position': center,
                            'is_inside': True,
                            'status': 'protected',
                            'first_seen': self.current_time,
                            'last_seen': self.current_time,
                            'last_alert_time': 0,
                            'bbox': (x1, y1, x2, y2),
                            'movement_path': [center],
                            'confidence': conf
                        }
                        print(f"🛡️ New protected object: {object_type} (ID:{track_id})")
        
        # Clean up stale objects (not seen for 10 seconds)
        stale_ids = []
        for track_id, pobj in self.protected_objects.items():
            if self.current_time - pobj['last_seen'] > 10:
                stale_ids.append(track_id)
        
        for track_id in stale_ids:
            del self.protected_objects[track_id]
            if track_id in self.object_history:
                del self.object_history[track_id]
        
        return {
            'detected_objects': detected_objects,
            'protected_objects': self.protected_objects,
            'alerts': alerts,
            'frame_count': self.frame_count
        }
    
    def draw_polygon(self, frame, color=(0, 255, 0), thickness=2, fill_alpha=0.2):
        """Draw the protection polygon on the frame"""
        if len(self.protection_polygon) == 0:
            return frame
        
        overlay = frame.copy()
        polygon_np = np.array(self.protection_polygon, dtype=np.int32)
        
        if len(self.protection_polygon) >= 3:
            # Fill polygon with transparency
            cv2.fillPoly(overlay, [polygon_np], color)
            frame = cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0)
            
            # Draw polygon outline
            cv2.polylines(frame, [polygon_np], True, color, thickness)
        
        # Draw points
        for i, pt in enumerate(self.protection_polygon):
            cv2.circle(frame, tuple(map(int, pt)), 6, (255, 255, 255), -1)
            cv2.circle(frame, tuple(map(int, pt)), 6, color, 2)
            cv2.putText(frame, str(i+1), (int(pt[0]) + 10, int(pt[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw line to next point (preview when drawing)
        if len(self.protection_polygon) >= 1 and not self.polygon_complete:
            # Line from last point (will be updated to mouse pos in main loop)
            pass
        
        return frame

# -------------------------------------------------
# ARGUMENT PARSING
# -------------------------------------------------
import argparse
def parse_args():
    parser = argparse.ArgumentParser("Polygon Zone Asset Protection System")
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
def run_polygon_asset_protection():
    args = parse_args()
    RTSP_URLS = args.camera
    AREA_NAMES = args.area

    # ---------------- DEVICE ----------------
    assert torch.cuda.is_available(), "CUDA GPU REQUIRED"
    torch.backends.cudnn.benchmark = True
    
    print("\n" + "=" * 60)
    print("  POLYGON ZONE ASSET PROTECTION SYSTEM")
    print("  GPU OPTIMIZED + USER-DEFINED PROTECTION ZONE")
    print("=" * 60)
    
    print(f"\n>>> USING DEVICE: {DEVICE.upper()} <<<")
    print(f"Model: {MODEL_PATH}")
    print(f"RTSP Streams: {len(RTSP_URLS)}")
    print("\n=== WORKFLOW ===")
    print("1. Draw protection zone by clicking points on the frame")
    print("2. Right-click or press [C] to complete the polygon")
    print("3. Press [ENTER] to start detection")
    print("4. Objects inside the zone will be protected")
    print("5. Alert when any protected object leaves the zone")
    print("\n=== CONTROLS ===")
    print("[Left-Click] Add polygon point")
    print("[Right-Click] or [C] Complete polygon")
    print("[ENTER] Start detection")
    print("[R] Reset polygon and detection")
    print("[S] Save polygon")
    print("[ESC] Quit")
    print("\n=== COLORS ===")
    print("🟢 GREEN: Protection zone / Protected object inside")
    print("🔴 RED: Object left the zone (ALERT)")
    print("🟡 YELLOW: Movement path trail")
    print("\n")

    # ---------------- CREATE DETECTORS ----------------
    detectors = [
        PolygonAssetProtector(
            model_path=MODEL_PATH,
            confidence=CONFIDENCE,
            device=DEVICE
        )
        for _ in range(len(RTSP_URLS))
    ]
    
    # ---------------- CREATE STREAMS (using RobustRTSPReader) ----------------
    rtsp_configs = [
        RTSPConfig(
            decoding_mode=DecodingMode.AUTO,  # GPU if available, else CPU
            buffer_size=QUEUE_SIZE,
            infinite_reconnect=True,
            max_reconnect_delay=30.0,
            stale_frame_timeout=10.0,
        )
        for _ in RTSP_URLS
    ]
    
    streams = [
        RobustRTSPReader(url, config=cfg).start()
        for url, cfg in zip(RTSP_URLS, rtsp_configs)
    ]
    
    print(f"\n✓ Initialized {len(streams)} robust RTSP streams (GPU/CPU auto-detect)")
    
    # ---------------- MOUSE STATE ----------------
    mouse_state = [{
        'current_pos': None,
        'drawing': True  # Start in drawing mode
    } for _ in range(len(RTSP_URLS))]
    
    def mouse_callback(event, x, y, flags, cam_id):
        detector = detectors[cam_id]
        state = mouse_state[cam_id]
        
        # Update current mouse position
        state['current_pos'] = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left-click: Add point to polygon
            if not detector.polygon_complete and not detector.detection_active:
                detector.add_polygon_point((x, y))
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click: Complete polygon
            if not detector.polygon_complete and len(detector.protection_polygon) >= 3:
                detector.complete_polygon()
    
    # ---------------- CREATE WINDOWS ----------------
    for i in range(len(RTSP_URLS)):
        cv2.namedWindow(AREA_NAMES[i])
        cv2.setMouseCallback(AREA_NAMES[i], mouse_callback, i)
    
    prev_time = time.time()
    
    # ---------------- MAIN LOOP ----------------
    try:
        while True:
            frames = [s.get_frame() for s in streams]
            if any(f is None for f in frames):
                # Show connection status
                for i, (frame, stream) in enumerate(zip(frames, streams)):
                    if frame is None:
                        stats = stream.get_stats()
                        print(f"[CAM-{i+1}] Waiting... State={stats.current_state.value}, Reconnects={stats.reconnect_count}")
                time.sleep(0.1)
                continue
            
            now = time.time()
            fps = 1.0 / max(now - prev_time, 0.0001)
            prev_time = now
            
            # Process each camera
            for i, (frame, detector) in enumerate(zip(frames, detectors)):
                display_frame = frame.copy()
                
                # Process detection
                result = detector.process(frame)
                
                # Determine polygon color based on state
                if detector.detection_active:
                    # Check if any object has left zone
                    has_alert = any(obj['status'] == 'left_zone' 
                                   for obj in result['protected_objects'].values())
                    polygon_color = (0, 0, 255) if has_alert else (0, 255, 0)
                else:
                    polygon_color = (255, 200, 0)  # Cyan/blue when drawing
                
                # Draw the protection polygon
                display_frame = detector.draw_polygon(display_frame, polygon_color)
                
                # Draw preview line when drawing polygon
                if not detector.polygon_complete and len(detector.protection_polygon) > 0:
                    last_pt = detector.protection_polygon[-1]
                    if mouse_state[i]['current_pos']:
                        cv2.line(display_frame, 
                                tuple(map(int, last_pt)), 
                                mouse_state[i]['current_pos'],
                                (200, 200, 200), 1, cv2.LINE_AA)
                
                # Draw detected objects and their paths
                for obj in result['detected_objects']:
                    b = obj['bbox']
                    track_id = obj['track_id']
                    is_inside = obj['is_inside']
                    
                    # Determine color based on status
                    if track_id in result['protected_objects']:
                        pobj = result['protected_objects'][track_id]
                        if pobj['status'] == 'left_zone':
                            # RED: Left the zone!
                            color = (0, 0, 255)
                            label = f"ALERT! {obj['object_type']}"
                            
                            # Draw movement path for this object
                            if 'movement_path' in pobj and len(pobj['movement_path']) > 1:
                                draw_movement_path(display_frame, pobj['movement_path'], (0, 255, 255))
                        else:
                            # GREEN: Protected and inside
                            color = (0, 255, 0)
                            label = f"Protected: {obj['object_type']}"
                    else:
                        # GRAY: Not protected (outside zone or not tracked)
                        color = (150, 150, 150)
                        label = f"{obj['object_type']}"
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (b[0], b[1]), (b[2], b[3]), color, 2)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(display_frame, 
                                 (b[0], b[1] - label_size[1] - 10),
                                 (b[0] + label_size[0] + 5, b[1]),
                                 color, -1)
                    cv2.putText(display_frame, label, (b[0] + 2, b[1] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw track ID
                    cv2.putText(display_frame, f"ID:{track_id}", (b[0], b[3] + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Send API alerts
                for alert in result['alerts']:
                    log_zone_alert(i, AREA_NAMES[i], alert, frame)
                
                # Draw HUD
                cv2.rectangle(display_frame, (10, 10), (400, 130), (0, 0, 0), -1)
                
                # Status info
                protected_count = len(result['protected_objects'])
                left_count = len([o for o in result['protected_objects'].values() 
                                 if o['status'] == 'left_zone'])
                
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if detector.detection_active:
                    cv2.putText(display_frame, "STATUS: MONITORING", (20, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Protected: {protected_count}", (20, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    if left_count > 0:
                        cv2.putText(display_frame, f"LEFT ZONE: {left_count}", (20, 110),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif detector.polygon_complete:
                    cv2.putText(display_frame, "STATUS: READY", (20, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(display_frame, "Press ENTER to start", (20, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(display_frame, "STATUS: DRAWING ZONE", (20, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
                    cv2.putText(display_frame, f"Points: {len(detector.protection_polygon)}", (20, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                    cv2.putText(display_frame, "Click to add, Right-click to complete", (20, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # Draw instructions at bottom
                cv2.rectangle(display_frame, (10, display_frame.shape[0] - 50), 
                             (500, display_frame.shape[0] - 10), (0, 0, 0), -1)
                
                if not detector.polygon_complete:
                    instructions = "Left-Click: Add Point | Right-Click: Complete Polygon"
                elif not detector.detection_active:
                    instructions = "Press ENTER to start | R: Reset"
                else:
                    instructions = "Monitoring... | R: Reset | ESC: Quit"
                
                cv2.putText(display_frame, instructions, 
                           (20, display_frame.shape[0] - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow(AREA_NAMES[i], display_frame)
            
            # Key controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER - Start detection
                for detector in detectors:
                    if detector.polygon_complete and not detector.detection_active:
                        detector.start_detection()
            
            elif key == ord('c') or key == ord('C'):  # Complete polygon
                for detector in detectors:
                    if not detector.polygon_complete:
                        detector.complete_polygon()
            
            elif key == ord('r') or key == ord('R'):  # Reset
                for detector in detectors:
                    detector.reset_polygon()
                print("All polygons reset!")
            
            elif key == ord('s') or key == ord('S'):  # Save polygon
                for detector in detectors:
                    detector.save_polygon()
                print("Polygons saved!")
            
            elif key == 27:  # ESC - Quit
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Save polygons on exit
        for detector in detectors:
            detector.save_polygon()
        
        # Cleanup
        for s in streams:
            s.stop()
        cv2.destroyAllWindows()
        print("Polygon asset protection system closed cleanly.")

# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    run_polygon_asset_protection()