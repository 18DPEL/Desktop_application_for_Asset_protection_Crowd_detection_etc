import cv2
import time
import torch
import numpy as np
import threading
from datetime import datetime
from ultralytics import YOLO
import requests
import base64

from RTSP_READER_CUDA.ffmpeg_rtsp import FFmpegCudaRTSP
from RTSP_READER_CUDA.inference import MicroBatchInferencer

# -------------------------------------------------
# HARD CONFIG
# -------------------------------------------------
MODEL_PATH = "yolov8n.pt"
DEVICE = "cuda"

FRAME_WIDTH = 960
FRAME_HEIGHT = 480
QUEUE_SIZE = 3
MICRO_BATCH_SIZE = 5

ALERT_CLASSES = ["person"]
CONF_THRESHOLD = 0.5

API_ENDPOINT = "http://192.168.1.7/cameradetection/public/api/intrusion_direction"
API_TIMEOUT = 5
API_INTERVAL = 5  # seconds per camera

CROWD_LIMIT = 1  # alerts if >=1 person

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
def log_intrusion(cam_id, area, alert_count, frame=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_status = "YES" if alert_count >= CROWD_LIMIT else "NO"
    print(f"[{timestamp}] CAM-{cam_id + 1} | Area={area} | Alert Count={alert_count} | Alert={alert_status}", flush=True)

    payload = {
        "timestamp": timestamp,
        "camera_id": cam_id + 1,
        "area": area,
        "alert_count": alert_count,
        "alert": alert_status
    }

    # Encode frame as base64 if provided
    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        payload["frame"] = frame_b64

    # API sending disabled
    # threading.Thread(target=send_to_server, args=(payload,), daemon=True).start()

# -------------------------------------------------
# ARGUMENT PARSING
# -------------------------------------------------
import argparse
def parse_args():
    parser = argparse.ArgumentParser("Intrusion Detection")
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
def run_intrusion_detection():
    args = parse_args()
    RTSP_URLS = args.camera
    AREA_NAMES = args.area

    # ---------------- DEVICE ----------------
    assert torch.cuda.is_available(), "CUDA GPU REQUIRED"
    torch.backends.cudnn.benchmark = True
    print(f"\n>>> USING DEVICE: {DEVICE.upper()} <<<\n")

    # ---------------- LOAD MODEL ----------------
    model = YOLO(MODEL_PATH).to(DEVICE)
    inferencer = MicroBatchInferencer(model, MICRO_BATCH_SIZE)

    # ---------------- STREAMS ----------------
    streams = [FFmpegCudaRTSP(url, FRAME_WIDTH, FRAME_HEIGHT, QUEUE_SIZE) for url in RTSP_URLS]

    # ---------------- ZONES ----------------
    zones = {
        i: {"points": [], "locked": False, "alert_count": 0, "last_sent": 0}
        for i in range(len(RTSP_URLS))
    }

    # ---------------- MOUSE CALLBACK ----------------
    def mouse_callback(event, x, y, flags, cam_id):
        zone = zones[cam_id]
        if zone["locked"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            zone["points"].append((x, y))
            print(f"[{AREA_NAMES[cam_id]}] Zone point added: {(x, y)}")

    # ---------------- WINDOWS ----------------
    for i in range(len(RTSP_URLS)):
        cv2.namedWindow(AREA_NAMES[i])
        cv2.setMouseCallback(AREA_NAMES[i], mouse_callback, i)

    prev_time = time.time()

    # ---------------- MAIN LOOP ----------------
    while True:
        frames = [s.read() for s in streams]
        if any(f is None for f in frames):
            continue

        now = time.time()
        fps = 1.0 / max(now - prev_time, 0.0001)
        prev_time = now

        with torch.inference_mode():
            results = inferencer.infer(frames)

        for i, (frame, res) in enumerate(zip(frames, results)):
            display = frame.copy()
            zone = zones[i]
            pts = zone["points"]
            alert_count = 0

            # Draw zone
            if len(pts) > 1:
                for j in range(len(pts)-1):
                    cv2.line(display, pts[j], pts[j+1], (0,255,255), 2)
                if zone["locked"]:
                    cv2.line(display, pts[-1], pts[0], (0,255,255), 2)

            # Detection logic
            if zone["locked"] and len(pts) >= 3:
                poly = np.array(pts, np.int32)
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    if cls_name not in ALERT_CLASSES:
                        continue
                    if box.conf[0] < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    inside = cv2.pointPolygonTest(poly, (cx, cy), False)
                    if inside >=0:
                        alert_count +=1
                        color=(0,0,255)
                        thickness=2
                    else:
                        color=(0,255,0)
                        thickness=1
                    cv2.rectangle(display, (x1,y1), (x2,y2), color, thickness)

                zone["alert_count"] = alert_count
                # Send API every API_INTERVAL seconds, include frame if alert
                if time.time() - zone["last_sent"] >= API_INTERVAL:
                    log_intrusion(i, AREA_NAMES[i], alert_count, frame=frame if alert_count>0 else None)
                    zone["last_sent"] = time.time()

                # HUD ALERT
                if alert_count > 0:
                    cv2.putText(display, f"ALERT! Persons in Zone: {alert_count}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255),3)
                else:
                    cv2.putText(display, "Zone Clear", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)

            # HUD
            cv2.putText(display, f"FPS: {fps:.1f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            if not zone["locked"]:
                cv2.putText(display, "Draw zone & press ENTER", (20,120), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
            else:
                cv2.putText(display, f"Detection Active | Alerts: {zone['alert_count']}", (20,120), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            cv2.imshow(AREA_NAMES[i], display)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            for i in zones:
                if not zones[i]["locked"] and len(zones[i]["points"]) >= 3:
                    zones[i]["locked"] = True
                    print(f"[{AREA_NAMES[i]}] Zone locked. Detection started.")
        elif key == ord('r'):  # RESET
            for i in zones:
                zones[i]["points"].clear()
                zones[i]["locked"] = False
                zones[i]["alert_count"] = 0
                print(f"[{AREA_NAMES[i]}] Zone reset.")
        elif key == 27:  # ESC
            break

    # Cleanup
    for s in streams:
        s.stop()
    cv2.destroyAllWindows()
    print("Intrusion detection closed cleanly.")

# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    run_intrusion_detection()