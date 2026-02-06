import cv2
import time
import torch
import numpy as np
import argparse
import requests
import threading
import json
import os
import base64
from datetime import datetime
from ultralytics import YOLO

# Optional imports for standalone mode - not needed for GUI
try:
    import supervision as sv
    from RTSP_READER_CUDA.ffmpeg_rtsp import FFmpegCudaRTSP
    from RTSP_READER_CUDA.inference import MicroBatchInferencer
    from config_file_2.config import (
        FRAME_HEIGHT as CFG_FRAME_HEIGHT,
        FRAME_WIDTH as CFG_FRAME_WIDTH,
        API_LOITERING_DETECTION as CFG_API_ENDPOINT,
        API_TIMEOUT as CFG_API_TIMEOUT,
        API_INTERVAL as CFG_API_INTERVAL,
        QUEUE_SIZE as CFG_QUEUE_SIZE,
        LOITER_TIME as CFG_LOITER_TIME,
        MICRO_BATCH_SIZE as CFG_MICRO_BATCH_SIZE,
    )
    STANDALONE_MODE = True
except ImportError as e:
    # This is expected when running from GUI (Asset_protection_gui.py)
    # The LoiteringDetector class works fine without these standalone imports
    print(f"ℹ️ Loitering Detection: GUI Integration Mode Enabled (Standalone modules skipped)")
    STANDALONE_MODE = False
    sv = None
    FFmpegCudaRTSP = None
    MicroBatchInferencer = None
# -------------------------------------------------
# HARD CONFIG
# -------------------------------------------------
MODEL_PATH = "yolov8n.pt"
DEVICE = "cuda"

FRAME_WIDTH = 660
FRAME_HEIGHT = 280
QUEUE_SIZE = 3
MICRO_BATCH_SIZE = 5

LOITER_TIME = 10# seconds to detect loitering

API_ENDPOINT = "http://192.168.1.14:8000/api/loitering_data"
API_TIMEOUT = 5  # seconds
API_INTERVAL = 5  # seconds per camera

# Frame skipping configuration (INTERNAL ONLY)
FRAME_SKIP = 5  # Process every 5th frame internally

# Polygon config file
POLYGON_CONFIG_FILE = r"D:\clean_puri_hall_project\Crowd_Detectio_n\loitering_detection_api_config.json"

# Auto-start configuration
AUTO_START_DETECTION = True  # Automatically start detection when polygon is loaded from JSON

# -------------------------------------------------
# FPS COUNTER CLASS
# -------------------------------------------------
class FPSCounter:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.prev_time = time.time()
        self.fps = 0.0
        
    def update(self):
        current_time = time.time()
        frame_time = current_time - self.prev_time
        self.prev_time = current_time
        
        # Add frame time to window
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        # Calculate average FPS
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / max(avg_frame_time, 0.0001)
        
        return self.fps

# -------------------------------------------------
# JSON FILE UTILITIES
# -------------------------------------------------
def ensure_json_file_exists(file_path):
    """Ensure the JSON file exists and has valid structure"""
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        initial_data = {"polygons": []}
        
        with open(file_path, 'w') as f:
            json.dump(initial_data, f, indent=2)
        print(f"✓ Created new loitering polygon config file: {file_path}")
        return True
    
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            if not content:
                print(f"⚠️ Config file is empty, recreating...")
                os.remove(file_path)
                return ensure_json_file_exists(file_path)
            
            data = json.loads(content)
            if 'polygons' not in data:
                data['polygons'] = []
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
            
    except json.JSONDecodeError:
        print(f"⚠️ Config file is corrupted, recreating...")
        os.remove(file_path)
        return ensure_json_file_exists(file_path)
    except Exception as e:
        print(f"⚠️ Error checking config file: {e}")
        return False

def load_json_safe(file_path):
    """Safely load JSON file with error handling"""
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def save_json_safe(file_path, data):
    """Safely save JSON file with NO backups"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✓ Saved to {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving to {file_path}: {e}")
        return False

# -------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------
def load_all_polygons(config_file=POLYGON_CONFIG_FILE):
    """Load all saved polygons from config file"""
    data = load_json_safe(config_file)
    if data is None:
        print(f"ℹ️ No polygon configuration file found or file is empty: {config_file}")
        return []
    
    polygons = data.get('polygons', [])
    print(f"\n📁 Loaded {len(polygons)} saved polygon(s) from {config_file}:")
    
    if polygons:
        for poly in polygons:
            camera_id = poly.get('camera_id', "unknown")
            area_name = poly.get('area_name', 'Unknown')
            points = len(poly.get('polygon', []))
            print(f"  Camera ID: {camera_id} ({area_name}): {points} points")
    else:
        print("  No polygons saved yet")
    
    return polygons

def print_polygon_statistics(config_file=POLYGON_CONFIG_FILE):
    """Print statistics about all saved polygons"""
    polygons = load_all_polygons(config_file)
    if not polygons:
        print("ℹ️ No polygon data available")
        return
    
    print(f"\n{'='*80}")
    print(f"LOITERING DETECTION POLYGON STATISTICS (File: {config_file})")
    print(f"{'='*80}")
    
    total_points = 0
    areas_by_camera = {}
    
    for poly in polygons:
        camera_id = poly.get('camera_id', "unknown")
        area_name = poly.get('area_name', 'Unknown')
        points = len(poly.get('polygon', []))
        
        total_points += points
        
        if camera_id not in areas_by_camera:
            areas_by_camera[camera_id] = {
                'areas': [],
                'total_points': 0
            }
        
        areas_by_camera[camera_id]['areas'].append({
            'name': area_name,
            'points': points
        })
        areas_by_camera[camera_id]['total_points'] += points
    
    print(f"Total Cameras: {len(areas_by_camera)}")
    print(f"Total Polygons: {len(polygons)}")
    print(f"Total Points: {total_points}")
    print(f"\nPer Camera Details:")
    
    for camera_id, data in areas_by_camera.items():
        print(f"\n  Camera ID: {camera_id}:")
        print(f"    Total Areas: {len(data['areas'])}")
        print(f"    Total Points: {data['total_points']}")
        
        for area in data['areas']:
            print(f"    • {area['name']}: {area['points']} points")
    
    print(f"{'='*80}\n")

# -------------------------------------------------
# ASYNC API SENDER
# -------------------------------------------------
def send_to_server(payload):
    try:
        requests.post(API_ENDPOINT, json=payload, timeout=API_TIMEOUT)
    except Exception as e:
        print(f"[API ERROR] {e}", flush=True)

# -------------------------------------------------
# LOGGING + API HELPER
# -------------------------------------------------
def log_loitering(camera_id, area, loitering_count, annotated_frame=None, original_frame=None):
    """Send loitering alert to API with annotated frame
    
    Args:
        camera_id: Custom camera ID from command line (e.g., "10", "12")
        area: Area name
        loitering_count: Number of loitering persons detected
        annotated_frame: Frame with annotations
        original_frame: Original frame without annotations
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Camera ID: {camera_id} | Area={area} | Loitering={loitering_count}", flush=True)
    payload = {
        "timestamp": timestamp,
        "camera_id": camera_id,  # Custom camera ID from command line
        "area": area,
        "loitering": loitering_count
    }
    
    # Encode annotated frame as base64 if provided, otherwise use original frame
    frame_to_send = annotated_frame if annotated_frame is not None else original_frame
    if frame_to_send is not None:
        _, buffer = cv2.imencode('.jpg', frame_to_send)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        payload["frame"] = frame_b64
        payload["frame_type"] = "annotated" if annotated_frame is not None else "original"
    
    # API sending disabled
    # threading.Thread(target=send_to_server, args=(payload,), daemon=True).start()

# -------------------------------------------------
# ZONE MANAGER CLASS
# -------------------------------------------------
class ZoneManager:
    def __init__(self, camera_id="0", area_name="", rtsp_url="", config_file=POLYGON_CONFIG_FILE):
        """Initialize ZoneManager with custom camera ID.
        
        Args:
            camera_id: Custom camera ID string from command line (e.g., "10", "12")
            area_name: Area name for this zone
            rtsp_url: RTSP stream URL
            config_file: Path to polygon configuration file
        """
        self.camera_id = camera_id  # Custom camera ID from command line (string like "10")
        self.area_name = area_name
        self.rtsp_url = rtsp_url
        self.config_file = config_file
        
        self.points = []
        self.locked = False
        self.person_data = {}
        self.last_sent = 0
        self.frame_counter = 0
        self.last_loitering_count = 0
        self.last_detections = []
        
        self.loaded_from_json = False
        
        ensure_json_file_exists(self.config_file)
        self.load_polygon()
    
    def load_polygon(self):
        """Load saved polygon from configuration file"""
        data = load_json_safe(self.config_file)
        if data is None:
            print(f"⚠️ Could not load polygon config file for Camera ID: {self.camera_id}")
            self.points = []
            self.locked = False
            self.loaded_from_json = False
            return
        
        polygons = data.get('polygons', [])
        for poly_data in polygons:
            if (poly_data.get('camera_id') == self.camera_id and 
                poly_data.get('area_name') == self.area_name):
                
                self.points = [tuple(p) for p in poly_data.get('polygon', [])]
                self.locked = len(self.points) >= 3
                self.loaded_from_json = self.locked
                
                if self.locked:
                    print(f"✓ Loaded polygon from JSON for Camera ID: {self.camera_id} ({self.area_name}) "
                          f"with {len(self.points)} points")
                return
        
        print(f"ℹ️ No saved polygon found for Camera ID: {self.camera_id} ({self.area_name})")
        self.points = []
        self.locked = False
        self.loaded_from_json = False
    
    def save_polygon(self):
        """Save polygon to configuration file with area information"""
        if len(self.points) < 3:
            print(f"⚠️ Camera ID: {self.camera_id}: Cannot save - polygon needs at least 3 points")
            return False
        
        data = load_json_safe(self.config_file)
        if data is None:
            data = {"polygons": []}
        
        polygon_data = {
            "camera_id": self.camera_id,
            "area_name": self.area_name,
            "polygon": [list(p) for p in self.points],
            "points_count": len(self.points),
            "frame_width": FRAME_WIDTH,
            "frame_height": FRAME_HEIGHT,
            "loiter_time": LOITER_TIME,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        existing_idx = -1
        for idx, poly in enumerate(data.get('polygons', [])):
            if (poly.get('camera_id') == self.camera_id and 
                poly.get('area_name') == self.area_name):
                existing_idx = idx
                break
        
        if existing_idx >= 0:
            data['polygons'][existing_idx] = polygon_data
            print(f"✓ Updated polygon for Camera ID: {self.camera_id} ({self.area_name})")
        else:
            data['polygons'].append(polygon_data)
            print(f"✓ Saved new polygon for Camera ID: {self.camera_id} ({self.area_name})")
        
        if save_json_safe(self.config_file, data):
            print(f"✓ Polygon coordinates saved for Camera ID: {self.camera_id} ({self.area_name})")
            print(f"  Points: {len(self.points)}")
            print(f"  File: {self.config_file}")
            return True
        else:
            return False
    
    def print_polygon_summary(self):
        """Print a summary of the polygon configuration"""
        if len(self.points) == 0:
            print(f"Camera ID: {self.camera_id} ({self.area_name}): No polygon defined")
            return
        
        print(f"\n{'='*60}")
        print(f"LOITERING POLYGON SUMMARY - Camera ID: {self.camera_id} ({self.area_name})")
        print(f"{'='*60}")
        print(f"Camera ID: {self.camera_id}")
        print(f"Status: {'LOCKED' if self.locked else 'UNLOCKED'}")
        print(f"Loaded from: {'JSON file' if self.loaded_from_json else 'Manual drawing'}")
        print(f"Points: {len(self.points)}")
        print(f"Detection: {'ACTIVE' if self.locked else 'INACTIVE'}")
        print(f"Loiter Time: {LOITER_TIME} seconds")
        print(f"Config File: {self.config_file}")
        
        for i, point in enumerate(self.points, 1):
            normalized_x = round(point[0] / FRAME_WIDTH, 4)
            normalized_y = round(point[1] / FRAME_HEIGHT, 4)
            print(f"  Point {i:2d}: ({point[0]:4d}, {point[1]:3d}) "
                  f"[Normalized: {normalized_x:.4f}, {normalized_y:.4f}]")
        
        if len(self.points) >= 3:
            polygon_np = np.array(self.points, dtype=np.int32)
            area = cv2.contourArea(polygon_np)
            area_percent = (area / (FRAME_WIDTH * FRAME_HEIGHT)) * 100
            print(f"\nPolygon Area: {area:.0f} pixels ({area_percent:.1f}% of frame)")
            print(f"Frame Size: {FRAME_WIDTH} x {FRAME_HEIGHT}")
        
        print(f"{'='*60}\n")
    
    def add_point(self, point):
        """Add a point to the polygon"""
        self.points.append(point)
        self.loaded_from_json = False
        print(f"📍 Camera ID: {self.camera_id}: Added point {len(self.points)}: {point}")
        
        if len(self.points) == 3:
            print(f"💡 Camera ID: {self.camera_id}: You have 3 points. Press ENTER to lock polygon!")
    
    def lock_polygon(self):
        """Lock the polygon (complete the shape)"""
        if len(self.points) >= 3:
            self.locked = True
            self.frame_counter = 0
            self.last_detections = []
            self.last_loitering_count = 0
            self.person_data = {}
            
            if self.save_polygon():
                print(f"✅ Camera ID: {self.camera_id}: Polygon locked with {len(self.points)} points!")
                print(f"▶️ Camera ID: {self.camera_id}: Loitering detection started!")
                self.print_polygon_summary()
            return True
        else:
            print(f"⚠️ Camera ID: {self.camera_id}: Need at least 3 points to lock polygon")
            return False
    
    def reset_polygon(self):
        """Reset the polygon"""
        self.points = []
        self.locked = False
        self.loaded_from_json = False
        self.last_loitering_count = 0
        self.frame_counter = 0
        self.last_detections = []
        self.person_data = {}
        print(f"🔄 Camera ID: {self.camera_id}: Polygon reset")
        print(f"💡 Camera ID: {self.camera_id}: Draw new polygon and press ENTER to start")
    
    def should_process_frame(self):
        """INTERNAL: Check if current frame should be processed based on FRAME_SKIP"""
        if not self.locked:
            return True
        
        self.frame_counter += 1
        if self.frame_counter >= FRAME_SKIP:
            self.frame_counter = 0
            return True
        return False

# -------------------------------------------------
# ARGUMENT PARSING
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Loitering Detection")
    
    parser.add_argument("--stats", action="store_true", 
                       help="Show polygon statistics and exit")
    parser.add_argument("--fix-json", action="store_true",
                       help="Fix corrupted JSON config file and exit")
    parser.add_argument("--list-polygons", action="store_true",
                       help="List all saved polygons and exit")
    
    parser.add_argument("--mode", choices=["single", "multi"], required=True)
    parser.add_argument("--camera", action="append", required=True)
    parser.add_argument("--camera_id", action="append", required=True,
                       help="Camera ID (like '10', '12', etc.)")
    parser.add_argument("--area", action="append", required=True)
    parser.add_argument("--no-auto-start", action="store_true",
                       help="Disable auto-start of detection when polygon loaded from JSON")
    parser.add_argument("--config-file", type=str, default=POLYGON_CONFIG_FILE,
                       help=f"Path to polygon configuration file (default: {POLYGON_CONFIG_FILE})")
    
    args = parser.parse_args()

    if len(args.camera) != len(args.area):
        raise ValueError("Each --camera must have a matching --area")
    if len(args.camera) != len(args.camera_id):
        raise ValueError("Each --camera must have a matching --camera_id")
    if args.mode == "single" and len(args.camera) != 1:
        raise ValueError("Single mode requires exactly one camera")
    
    # Check for duplicate camera IDs
    if len(set(args.camera_id)) != len(args.camera_id):
        raise ValueError("Duplicate camera IDs found. Each camera must have a unique ID")
    
    return args

# -------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------
def run_loitering_app():
    global AUTO_START_DETECTION, POLYGON_CONFIG_FILE
    
    args = parse_args()
    
    # Handle special commands first
    if args.fix_json:
        print("🛠️ Fixing JSON configuration file...")
        if ensure_json_file_exists(args.config_file):
            print("✅ JSON file fixed successfully")
            data = load_json_safe(args.config_file)
            if data:
                print(f"\nCurrent contents:")
                print(f"Total polygons: {len(data.get('polygons', []))}")
                for poly in data.get('polygons', []):
                    camera_id = poly.get('camera_id', "unknown")
                    area = poly.get('area_name', 'Unknown')
                    points = len(poly.get('polygon', []))
                    print(f"  Camera ID: {camera_id} ({area}): {points} points")
        return
    
    if args.stats:
        print_polygon_statistics(args.config_file)
        return
    
    if args.list_polygons:
        polygons = load_all_polygons(args.config_file)
        if not polygons:
            print(f"No polygons saved yet in {args.config_file}.")
        return
    
    RTSP_URLS = args.camera
    AREA_NAMES = args.area
    CAMERA_IDS = args.camera_id  # Custom camera IDs from command line (like ["10", "11"])
    
    if args.no_auto_start:
        AUTO_START_DETECTION = False
        print("⚠️ Auto-start disabled. Always press ENTER to start detection.")
    
    if args.config_file != POLYGON_CONFIG_FILE:
        POLYGON_CONFIG_FILE = args.config_file
        print(f"Using custom config file: {POLYGON_CONFIG_FILE}")
    
    config_file = args.config_file
    
    # ---------------- DEVICE ----------------
    assert torch.cuda.is_available(), "CUDA GPU REQUIRED"
    torch.backends.cudnn.benchmark = True
    
    print("\n" + "=" * 60)
    print("  LOITERING DETECTION SYSTEM")
    print("  GPU OPTIMIZED + INTERNAL FRAME SKIPPING")
    print("=" * 60)
    
    print(f"\n>>> USING DEVICE: {DEVICE.upper()} <<<")
    print(f"Model: {MODEL_PATH}")
    print(f"RTSP Streams: {len(RTSP_URLS)}")
    print(f"Loiter Time: {LOITER_TIME} seconds")
    print(f"Internal Frame Skip: 1/{FRAME_SKIP} (TRANSPARENT TO USER)")
    print(f"Auto-start (JSON load): {'ENABLED' if AUTO_START_DETECTION else 'DISABLED'}")
    
    # Print camera configuration
    print(f"\n📷 CAMERA CONFIGURATION:")
    for i, (camera_id, area_name) in enumerate(zip(CAMERA_IDS, AREA_NAMES)):
        print(f"  Stream {i+1}: Camera ID: {camera_id}, Area: {area_name}")
    print(f"Config File: {config_file}")
    
    print(f"\n🔧 Checking configuration file...")
    if ensure_json_file_exists(config_file):
        print("✅ Configuration file ready")
    else:
        print("⚠️ Could not create configuration file, using default settings")
    
    print("\n=== WORKFLOW ===")
    print("1. If polygon exists in JSON → Detection starts automatically")
    print("2. If no saved polygon → Draw loitering zone by clicking points")
    print("3. Press ENTER to lock polygon and start detection")
    print("4. Alert when person stays in zone for >5 seconds")
    
    print("\n=== RULES ===")
    print("✓ Polygon from JSON → Auto-start (no ENTER needed)")
    print("✓ New polygon → Press ENTER to start")
    print("✓ Reset polygon → Press ENTER to start again")
    
    print("\n=== CONTROLS ===")
    print("[Left-Click] Add polygon point")
    print("[ENTER] Lock polygon and start detection")
    print("[R] Reset polygon")
    print("[S] Save polygon")
    print("[P] Print polygon coordinates")
    print("[ESC] Quit")
    
    print("\n=== COLORS ===")
    print("🟢 GREEN: Person in zone (<5 seconds)")
    print("🔴 RED: Person in zone (>5 seconds) = LOITERING")
    print("⚫ GRAY: Person outside zone")
    print("🟡 YELLOW: Polygon outline")
    print("")

    # ---------------- LOAD MODEL ----------------
    model = YOLO(MODEL_PATH).to(DEVICE)
    tracker = sv.ByteTrack()
    inferencer = MicroBatchInferencer(model, MICRO_BATCH_SIZE)

    # ---------------- ZONE MANAGERS ----------------
    zone_managers = [
        ZoneManager(
            camera_id=CAMERA_IDS[i],  # Use custom camera ID from command line
            area_name=AREA_NAMES[i],
            rtsp_url=RTSP_URLS[i],
            config_file=config_file
        )
        for i in range(len(RTSP_URLS))
    ]
    
    # Load existing polygons and check auto-start status
    load_all_polygons(config_file)
    
    # Check which cameras have auto-started
    auto_started_cameras = []
    for i, zone_manager in enumerate(zone_managers):
        if zone_manager.locked and AUTO_START_DETECTION and zone_manager.loaded_from_json:
            auto_started_cameras.append(zone_manager.camera_id)
    
    # Startup status
    print("\n" + "="*60)
    print("STARTUP STATUS")
    print("="*60)
    
    polygons_loaded = False
    for i, zone_manager in enumerate(zone_managers):
        if zone_manager.locked:
            print(f"✅ Camera ID: {zone_manager.camera_id}: Polygon loaded from JSON - Detection STARTED")
            polygons_loaded = True
        else:
            print(f"📝 Camera ID: {zone_manager.camera_id}: No saved polygon - Please DRAW polygon")
    
    if auto_started_cameras:
        print(f"\n✅ Auto-started loitering detection for Camera ID(s): {', '.join(auto_started_cameras)}")
    
    if not polygons_loaded:
        print("\n" + "="*60)
        print("FIRST TIME SETUP INSTRUCTIONS:")
        print("="*60)
        print("1. Click on the camera window to draw polygon points")
        print("2. Click at least 3 points to define loitering zone")
        print("3. Press ENTER to lock polygon and start detection")
        print("4. Polygon will be saved automatically for next run")
        print("="*60)

    # ---------------- STREAMS ----------------
    streams = [FFmpegCudaRTSP(url, FRAME_WIDTH, FRAME_HEIGHT, QUEUE_SIZE) for url in RTSP_URLS]

    # ---------------- MOUSE CALLBACK ----------------
    def mouse_callback(event, x, y, flags, camera_idx):
        zone_manager = zone_managers[camera_idx]
        if zone_manager.locked:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            zone_manager.add_point((x, y))

    # ---------------- WINDOWS ----------------
    for i in range(len(RTSP_URLS)):
        window_title = f"Camera ID: {CAMERA_IDS[i]} - {AREA_NAMES[i]}"
        cv2.namedWindow(window_title)
        cv2.setMouseCallback(window_title, mouse_callback, i)

    # ---------------- FPS COUNTERS ----------------
    fps_counters = [FPSCounter(window_size=30) for _ in range(len(RTSP_URLS))]

    # ---------------- MAIN LOOP ----------------
    try:
        while True:
            frames = [s.read() for s in streams]
            if any(f is None for f in frames):
                continue

            # Prepare frames for inference (INTERNAL frame skipping logic)
            frames_to_process = []
            indices_to_process = []
            
            for i, (frame, zone_manager) in enumerate(zip(frames, zone_managers)):
                if zone_manager.should_process_frame():
                    frames_to_process.append(frame)
                    indices_to_process.append(i)
            
            # Perform inference only on selected frames
            if frames_to_process:
                with torch.inference_mode():
                    results = inferencer.infer(frames_to_process)
            else:
                results = []

            # Process each camera
            for i, frame in enumerate(frames):
                display = frame.copy()
                zone_manager = zone_managers[i]
                fps_counter = fps_counters[i]
                
                # Update FPS for this camera
                fps = fps_counter.update()
                
                pts = zone_manager.points
                
                # Determine polygon color based on state
                if zone_manager.locked:
                    polygon_color = (0, 255, 0)  # Green when locked
                else:
                    polygon_color = (0, 255, 255)  # Yellow when drawing
                
                # Draw zone (always visible)
                if len(pts) > 1:
                    for j in range(len(pts)-1):
                        cv2.line(display, pts[j], pts[j+1], polygon_color, 2)
                    if zone_manager.locked:
                        cv2.line(display, pts[-1], pts[0], polygon_color, 2)
                
                # Draw polygon points
                for j, pt in enumerate(pts):
                    cv2.circle(display, pt, 6, (255, 255, 255), -1)
                    cv2.circle(display, pt, 6, polygon_color, 2)
                    if not zone_manager.locked:
                        cv2.putText(display, str(j+1), (pt[0] + 10, pt[1] - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, polygon_color, 2)
                
                loitering_count = 0
                
                # Check if this camera's frame was processed internally
                if i in indices_to_process:
                    idx_in_results = indices_to_process.index(i)
                    
                    if zone_manager.locked and len(pts) >= 3 and idx_in_results < len(results):
                        res = results[idx_in_results]
                        poly = np.array(pts, np.int32)
                        
                        current_detections = []
                        
                        detections = sv.Detections.from_ultralytics(res)
                        detections = tracker.update_with_detections(detections)
                        
                        person_idx = [idx for idx, cid in enumerate(detections.class_id) if cid == 0]
                        
                        if person_idx:
                            person_xyxy = detections.xyxy[person_idx]
                            person_ids = detections.tracker_id[person_idx]
                        else:
                            person_xyxy = []
                            person_ids = []
                        
                        current_time = time.time()
                        current_ids_in_zone = set()
                        
                        for xyxy, track_id in zip(person_xyxy, person_ids):
                            x1, y1, x2, y2 = map(int, xyxy)
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                            inside = cv2.pointPolygonTest(poly, (cx, cy), False) >= 0

                            if inside:
                                current_ids_in_zone.add(track_id)
                                if track_id not in zone_manager.person_data:
                                    zone_manager.person_data[track_id] = {
                                        "start_time": current_time, 
                                        "last_seen": current_time, 
                                        "loitering": False
                                    }
                                else:
                                    zone_manager.person_data[track_id]["last_seen"] = current_time

                                elapsed = current_time - zone_manager.person_data[track_id]["start_time"]
                                if elapsed >= LOITER_TIME:
                                    zone_manager.person_data[track_id]["loitering"] = True
                                    loitering_count += 1
                                    color = (0, 0, 255)  # Red for loitering
                                    text = f"LOITER {elapsed:.0f}s"
                                else:
                                    color = (0, 255, 0)  # Green for in zone but not loitering
                                    text = f"ID:{track_id} ({elapsed:.0f}s)"
                            else:
                                color = (100, 100, 100)  # Gray for outside zone
                                text = f"ID:{track_id}"
                            
                            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display, text, (x1, max(y1 - 10, 20)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            cv2.circle(display, (cx, cy), 3, color, -1)
                            
                            current_detections.append({
                                'xyxy': (x1, y1, x2, y2),
                                'center': (cx, cy),
                                'color': color,
                                'text': text,
                                'track_id': track_id,
                                'inside': inside
                            })
                        
                        zone_manager.last_detections = current_detections
                        
                        for tid in list(zone_manager.person_data.keys()):
                            data = zone_manager.person_data[tid]
                            if tid not in current_ids_in_zone and (current_time - data["last_seen"] > 2):
                                del zone_manager.person_data[tid]
                        
                        zone_manager.last_loitering_count = loitering_count
                        
                        # Store that we need to send API (will be done after all annotations)
                        zone_manager.should_send_api = True
                        zone_manager.api_loitering_count = loitering_count
                    
                    elif not zone_manager.locked and idx_in_results < len(results):
                        res = results[idx_in_results]
                        detections = sv.Detections.from_ultralytics(res)
                        person_idx = [idx for idx, cid in enumerate(detections.class_id) if cid == 0]
                        
                        if person_idx:
                            person_xyxy = detections.xyxy[person_idx]
                            
                            for xyxy in person_xyxy:
                                x1, y1, x2, y2 = map(int, xyxy)
                                color = (0, 255, 0)
                                cv2.rectangle(display, (x1, y1), (x2, y2), color, 1)
                                cv2.putText(display, "Person", (x1, max(y1 - 10, 20)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                else:
                    if zone_manager.locked and len(pts) >= 3:
                        for detection in zone_manager.last_detections:
                            x1, y1, x2, y2 = detection['xyxy']
                            cx, cy = detection['center']
                            color = detection['color']
                            text = detection['text']
                            
                            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display, text, (x1, max(y1 - 10, 20)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            cv2.circle(display, (cx, cy), 3, color, -1)
                        
                        loitering_count = zone_manager.last_loitering_count
                
                # ---------------- HUD ----------------
                cv2.putText(display, f"FPS: {fps:.1f}", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.putText(display, f"Camera ID: {zone_manager.camera_id}: {zone_manager.area_name}", (20, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                if zone_manager.locked:
                    if zone_manager.loaded_from_json:
                        source_text = "Loaded from JSON"
                        source_color = (0, 255, 255)
                    else:
                        source_text = "Manually drawn"
                        source_color = (255, 255, 0)
                    
                    cv2.putText(display, source_text, (20, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, source_color, 1)
                    
                    if loitering_count > 0:
                        cv2.putText(display, f"ALERT! Loitering: {loitering_count}", 
                                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        cv2.putText(display, "LOITERING DETECTED!", (20, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    else:
                        cv2.putText(display, "Zone Clear", (20, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display, "No loitering", (20, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
                    persons_in_zone = sum(1 for d in zone_manager.last_detections if d.get('inside', False))
                    cv2.putText(display, f"Persons in zone: {persons_in_zone}", (20, 180), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    cv2.putText(display, f"Polygon: {len(pts)} points", (20, 195),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    if len(pts) >= 3:
                        cv2.putText(display, "Ready to lock!", (20, 95),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        cv2.putText(display, "Press ENTER to start", (20, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    else:
                        cv2.putText(display, "Draw polygon (min 3 points)", (20, 95),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    
                    if len(pts) > 0:
                        cv2.putText(display, f"Points: {len(pts)}/3", (20, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                if not zone_manager.locked:
                    if len(pts) >= 3:
                        controls = "ENTER: Lock & Start | R: Reset | S: Save | P: Print"
                    else:
                        controls = "Left-Click: Add points | R: Reset"
                else:
                    controls = "Monitoring... | R: Reset | S: Save | P: Print | ESC: Quit"
                
                cv2.putText(display, controls, 
                           (10, display.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # ---- LOG + API SEND (after all annotations are drawn) ----
                if zone_manager.locked and hasattr(zone_manager, 'should_send_api') and zone_manager.should_send_api:
                    now_ts = time.time()
                    if now_ts - zone_manager.last_sent >= API_INTERVAL:
                        log_loitering(
                            zone_manager.camera_id, zone_manager.area_name, 
                            getattr(zone_manager, 'api_loitering_count', zone_manager.last_loitering_count),
                            annotated_frame=display,
                            original_frame=frame
                        )
                        zone_manager.last_sent = now_ts
                    zone_manager.should_send_api = False
                
                window_title = f"Camera ID: {zone_manager.camera_id} - {zone_manager.area_name}"
                cv2.imshow(window_title, display)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER
                for zone_manager in zone_managers:
                    if not zone_manager.locked and len(zone_manager.points) >= 3:
                        zone_manager.lock_polygon()
            elif key == ord('r') or key == ord('R'):  # RESET
                for zone_manager in zone_managers:
                    zone_manager.reset_polygon()
                print("All polygons reset!")
            elif key == ord('s') or key == ord('S'):  # Save polygon
                for zone_manager in zone_managers:
                    if len(zone_manager.points) >= 3:
                        zone_manager.save_polygon()
                print("Polygons saved!")
            elif key == ord('p') or key == ord('P'):  # Print polygon summary
                for zone_manager in zone_managers:
                    zone_manager.print_polygon_summary()
            elif key == 27:  # ESC
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\n" + "="*60)
        print("SAVING POLYGON COORDINATES...")
        print("="*60)
        
        for zone_manager in zone_managers:
            if len(zone_manager.points) >= 3:
                zone_manager.save_polygon()
                zone_manager.print_polygon_summary()
        
        print("\n" + "="*60)
        print("FINAL LOITERING POLYGON SUMMARY")
        print("="*60)
        print_polygon_statistics(config_file)
        
        for s in streams:
            s.stop()
        cv2.destroyAllWindows()
        print("✅ Loitering detection system closed cleanly.")


# -------------------------------------------------
# GUI-COMPATIBLE LOITERING DETECTOR CLASS
# (For use with Asset_protection_gui.py)
# -------------------------------------------------
class LoiteringDetector:
    """
    Loitering Detection for GUI integration.
    Tracks persons in a defined polygon zone and detects loitering.
    Same logic as ZoneManager but simplified for GUI popup use.
    """
    
    def __init__(self, model_path=MODEL_PATH, confidence=0.5, device=DEVICE,
                 loiter_time=LOITER_TIME):
        self.confidence = confidence
        self.device = device
        self.loiter_time = loiter_time
        
        # Initialize YOLO model
        try:
            self.model = YOLO(model_path).to(device)
            print(f"✓ LoiteringDetector initialized on {device}")
        except Exception as e:
            self.model = None
            print(f"⚠️ LoiteringDetector model init failed: {e}")
        
        # Polygon region
        self.protection_polygon = []
        self.polygon_complete = False
        self.detection_active = False
        
        # Person tracking (same logic as ZoneManager)
        self.person_data = {}  # track_id: {start_time, last_seen, loitering}
        self.loitering_count = 0
        
        # Frame state
        self.frame_count = 0
        self.current_time = time.time()
    
    def is_inside_polygon(self, point):
        """Check if a point is inside the protection polygon"""
        if len(self.protection_polygon) < 3:
            return False
        
        polygon_np = np.array(self.protection_polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(polygon_np, point, False)
        return result >= 0
    
    def start_detection(self):
        """Start detection after polygon is defined"""
        if not self.polygon_complete:
            print("⚠️ Cannot start detection - polygon not complete!")
            return False
        
        self.detection_active = True
        self.person_data = {}
        self.loitering_count = 0
        print(f"▶️ Loitering Detection started! Loiter time: {self.loiter_time}s")
        return True
    
    def stop_detection(self):
        """Stop detection"""
        self.detection_active = False
        print("⏹️ Loitering Detection stopped")
    
    def reset_polygon(self):
        """Reset the polygon and detection state"""
        self.protection_polygon = []
        self.polygon_complete = False
        self.detection_active = False
        self.person_data = {}
        self.loitering_count = 0
        print("🔄 Loitering zone reset")
    
    def process(self, frame):
        """Process a single frame for loitering detection
        
        Returns dict with:
            - detected_objects: list of objects with is_loitering, time_in_zone
            - loitering_count: number of persons loitering
            - alerts: list of alert dicts
        """
        self.current_time = time.time()
        self.frame_count += 1
        
        alerts = []
        detected_objects = []
        
        if not self.detection_active:
            return {
                'detected_objects': [],
                'loitering_count': 0,
                'alerts': [],
                'frame_count': self.frame_count
            }
        
        if self.model is None:
            return {
                'detected_objects': [],
                'loitering_count': 0,
                'alerts': [],
                'frame_count': self.frame_count
            }
        
        # YOLO detection with tracking
        results = self.model.track(frame, persist=True, verbose=False, conf=self.confidence)
        
        current_ids_in_zone = set()
        loitering_count = 0
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            has_ids = boxes.id is not None
            
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                
                # Only track persons (class 0)
                if cls_id != 0:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                track_id = int(boxes.id[i]) if has_ids else i
                conf = float(box.conf[0])
                
                # Check if person is inside polygon
                is_inside = self.is_inside_polygon(center)
                
                obj_data = {
                    'track_id': track_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'class_id': cls_id,
                    'object_type': 'person',
                    'confidence': conf,
                    'is_inside': is_inside,
                    'is_loitering': False,
                    'time_in_zone': 0
                }
                
                if is_inside:
                    current_ids_in_zone.add(track_id)
                    
                    if track_id not in self.person_data:
                        # New person entering zone - start timing
                        self.person_data[track_id] = {
                            'start_time': self.current_time,
                            'last_seen': self.current_time,
                            'loitering': False
                        }
                    else:
                        self.person_data[track_id]['last_seen'] = self.current_time
                    
                    # Calculate time in zone
                    elapsed = self.current_time - self.person_data[track_id]['start_time']
                    obj_data['time_in_zone'] = elapsed
                    
                    # Check if loitering (stayed > loiter_time)
                    if elapsed >= self.loiter_time:
                        self.person_data[track_id]['loitering'] = True
                        obj_data['is_loitering'] = True
                        loitering_count += 1
                        
                        # Generate alert
                        alerts.append({
                            'alert_type': 'LOITERING',
                            'track_id': track_id,
                            'object_type': 'person',
                            'time_in_zone': elapsed,
                            'position': center
                        })
                
                detected_objects.append(obj_data)
        
        # Clean up persons who left the zone (not seen for > 2 seconds)
        for tid in list(self.person_data.keys()):
            if tid not in current_ids_in_zone:
                if self.current_time - self.person_data[tid]['last_seen'] > 2:
                    del self.person_data[tid]
        
        self.loitering_count = loitering_count
        
        return {
            'detected_objects': detected_objects,
            'loitering_count': loitering_count,
            'alerts': alerts,
            'frame_count': self.frame_count
        }


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    run_loitering_app()