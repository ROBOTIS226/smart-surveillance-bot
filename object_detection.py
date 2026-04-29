import cv2
from ultralytics import YOLO
import time
import requests
import argparse
import os
import sys

# --- CONFIGURATION (STATION MODE) ---
# The ESP32 camera should be in "station" mode with an IP from your router (DHCP).
# Provide the IP as an argument (--ip), via ESP_CAM_IP env var, or enter it interactively when prompted.
DEFAULT_IP = "192.168.137.179"  # <-- update to your camera's station IP if you know it

parser = argparse.ArgumentParser(description="ESP32-CAM YOLO viewer (station mode).")
parser.add_argument("--ip", "-i", help="ESP32-CAM IP (e.g. 192.168.1.30)")
args = parser.parse_args()

ip_candidate = args.ip or os.getenv("ESP_CAM_IP")
if not ip_candidate:
    try:
        ip_candidate = input(f"Enter ESP32-CAM station IP [{DEFAULT_IP}]: ").strip() or DEFAULT_IP
    except Exception:
        # Non-interactive environment: fall back to default
        ip_candidate = DEFAULT_IP

# normalize to include scheme
if not ip_candidate.startswith("http://") and not ip_candidate.startswith("https://"):
    ESP_IP = f"http://{ip_candidate}"
else:
    ESP_IP = ip_candidate

STREAM_URL = f"{ESP_IP}/stream"
HANDSHAKE_URL = f"{ESP_IP}/handshake"

model = YOLO("yolov8n.pt")


def try_handshake():
    try:
        print(f"🔗 Sending handshake to {HANDSHAKE_URL} ...")
        r = requests.get(HANDSHAKE_URL, timeout=3)

        if r.ok and r.text.strip() == "OK":
            print("✅ Handshake successful")
            return True
        else:
            print("❌ Handshake rejected or unexpected response:", r.text if r is not None else "No response")
            return False

    except Exception as e:
        print("⚠ Handshake error:", e)
        return False


def connect_stream():
    print(f"🎥 Connecting to ESP32 stream at {STREAM_URL} ...")
    cap = cv2.VideoCapture(STREAM_URL)
    time.sleep(1)

    if not cap.isOpened():
        print("❌ Could not open stream")
        return None

    print("✅ Stream connected")
    return cap


def draw_boxes(frame, results):
    for obj in results[0].boxes:
        x1, y1, x2, y2 = map(int, obj.xyxy[0])
        conf = float(obj.conf[0]) if hasattr(obj, "conf") else 0.0
        cls = int(obj.cls[0]) if hasattr(obj, "cls") else 0
        label = f"{model.names[cls]} {conf:.2f}" if model and hasattr(model, "names") else ""
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if label:
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main():
    print(f"Running in STATION mode. Target camera: {ESP_IP}")
    while True:
        time.sleep(1)
        if not try_handshake():
            print("🔄 Retrying handshake in 3 seconds...\n")
            time.sleep(3)
            continue

        cap = connect_stream()
        if cap is None:
            time.sleep(3)
            continue

        try:
            while True:
                time.sleep(0.05)
                ret, frame = cap.read()

                if not ret or frame is None:
                    print("⚠ Stream lost. Reconnecting...")
                    break

                results = model(frame, verbose=False)
                draw_boxes(frame, results)

                cv2.imshow("YOLO Detection (ESP32-CAM - Station)", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

        except KeyboardInterrupt:
            print("👋 Exit requested by user")
            break

        except Exception as e:
            print("❌ Unexpected error:", e)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("♻ Restarting process...\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)