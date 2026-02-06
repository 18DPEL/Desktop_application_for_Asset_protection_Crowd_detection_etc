import cv2
import time
import torch
import numpy as np
import argparse
import requests
import threading
from datetime import datetime
from ultralytics import YOLO

from RTSP_READER_CUDA.ffmpeg_rtsp import FFmpegCudaRTSP
from RTSP_READER_CUDA.inference import MicroBatchInferencer

# -------------------------------------------------
# HARD CONFIG (LOCAL ONLY - NO CONFIG FILE)
# -------------------------------------------------
MODEL_PATH = "yolov8n.pt"
DEVICE = "cuda"

FRAME_WIDTH = 960
FRAME_HEIGHT = 480
QUEUE_SIZE = 3
MICRO_BATCH_SIZE = 5

CROWD_LIMIT = 10

API_ENDPOINT = "http://192.168.1.7/cameradetection/public/api/receive-data"
API_TIMEOUT = 5  # seconds
API_INTERVAL = 5  # seconds per camera

# -------------------------------------------------
# ASYNC API SENDER
# -------------------------------------------------
def send_to_server(payload):
    try:
        requests.post(
            API_ENDPOINT,
            json=payload,
            timeout=API_TIMEOUT
        )
    except Exception as e:
        print(f"[API ERROR] {e}", flush=True)

# -------------------------------------------------
# LOGGING + API HELPER
# -------------------------------------------------
def log_zone_status(cam_id, area, count, limit):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert = "YES" if count >= limit else "NO"

    # ---- CONSOLE LOG ----
    print(
        f"[{timestamp}] "
        f"CAM-{cam_id + 1} | "
        f"Area={area} | "
        f"Persons={count} | "
        f"Alert={alert}",
        flush=True
    )

    # ---- API PAYLOAD ----
    payload = {
        "timestamp": timestamp,
        "camera_id": cam_id + 1,
        "area": area,
        "persons": count,
        "crowd_limit": limit,
        "alert": alert
    }

    # ---- NON-BLOCKING SEND ---- (API sending disabled)
    # threading.Thread(
    #     target=send_to_server,
    #     args=(payload,),
    #     daemon=True
    # ).start()

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
def run_app():
    args = parse_args()

    RTSP_URLS = args.camera
    AREA_NAMES = args.area

    # ---------------- DEVICE ----------------
    assert torch.cuda.is_available(), "CUDA GPU REQUIRED"
    torch.backends.cudnn.benchmark = True
    print("\n>>> GPU MODE ENABLED <<<\n")

    # ---------------- LOAD MODEL ----------------
    model = YOLO(MODEL_PATH).to(DEVICE)

    # ---------------- STREAMS ----------------
    streams = [
        FFmpegCudaRTSP(
            url,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            QUEUE_SIZE
        )
        for url in RTSP_URLS
    ]

    inferencer = MicroBatchInferencer(
        model,
        MICRO_BATCH_SIZE
    )

    # ---------------- ZONES ----------------
    zones = {
        i: {
            "points": [],
            "locked": False,
            "area": AREA_NAMES[i],
            "last_sent": 0  # timestamp of last API send
        }
        for i in range(len(RTSP_URLS))
    }

    # ---------------- MOUSE CALLBACK ----------------
    def mouse_callback(event, x, y, flags, cam_id):
        zone = zones[cam_id]
        if zone["locked"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            zone["points"].append((x, y))
            print(f"[{zone['area']}] Zone point added: {(x, y)}")

    # ---------------- WINDOWS ----------------
    for i in range(len(RTSP_URLS)):
        cv2.namedWindow(zones[i]["area"])
        cv2.setMouseCallback(zones[i]["area"], mouse_callback, i)

    prev_time = time.time()

    # ---------------- MAIN LOOP ----------------
    while True:
        frames = []

        for s in streams:
            frame = s.read()
            if frame is None:
                break
            frames.append(frame)

        if len(frames) != len(streams):
            continue

        now = time.time()
        fps = 1.0 / max(now - prev_time, 0.0001)
        prev_time = now

        with torch.inference_mode():
            results = inferencer.infer(frames)

        for i, res in enumerate(results):
            frame = frames[i]
            zone = zones[i]
            pts = zone["points"]
            inside_count = 0

            # ---- DRAW ZONE ----
            if len(pts) > 1:
                for j in range(len(pts) - 1):
                    cv2.line(frame, pts[j], pts[j + 1], (0, 255, 255), 2)
                if zone["locked"]:
                    cv2.line(frame, pts[-1], pts[0], (0, 255, 255), 2)

            if zone["locked"] and len(pts) >= 3:
                poly = np.array(pts, np.int32)

                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    inside = cv2.pointPolygonTest(poly, (cx, cy), False)
                    color = (0, 0, 255) if inside >= 0 else (0, 255, 0)

                    if inside >= 0:
                        inside_count += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ---- HUD ----
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.putText(frame, f"Area: {zone['area']}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            if zone["locked"]:
                cv2.putText(frame, f"Persons: {inside_count}", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # ---- LOG + API SEND (every 5 seconds) ----
                now_ts = time.time()
                if now_ts - zone["last_sent"] >= API_INTERVAL:
                    log_zone_status(i, zone["area"], inside_count, CROWD_LIMIT)
                    zone["last_sent"] = now_ts

                if inside_count >= CROWD_LIMIT:
                    cv2.putText(frame, "CROWD ALERT!", (20, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Draw zone & press ENTER", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow(zone["area"], frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            for i in zones:
                if not zones[i]["locked"] and len(zones[i]["points"]) >= 3:
                    zones[i]["locked"] = True
                    print(f"[{zones[i]['area']}] Zone locked")
        elif key == 27:
            break

    for s in streams:
        s.stop()

    cv2.destroyAllWindows()
    print("Application closed cleanly.")

# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------
if __name__ == "__main__":
    run_app()