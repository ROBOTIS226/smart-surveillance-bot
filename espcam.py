import cv2
from ultralytics import YOLO
import time
import requests
import threading

# --- NETWORK CONFIGURATION (UPDATE THESE!) ---
# 1. Check Serial Monitor of Robot for this IP
ROBOT_IP = "http://192.168.137.200"  #esp32-280F78
# 2. Check Serial Monitor of Camera for this IP
CAM_IP   = "http://192.168.137.30"   #esp32-AC28D0wwwwwwwwssssssaa

# Command URLs
ROBOT_CMD_URL = f"{ROBOT_IP}/command"
CAM_STREAM_URL = f"{CAM_IP}/stream"
CAM_HANDSHAKE_URL = f"{CAM_IP}/handshake"

# --- GLOBAL STATE ---
last_command = "stop"
running = True # Global flag to stop threads when we quit
model = YOLO("yolov8n.pt")

# --- ROBOT CONTROL THREAD ---
def send_robot_command(direction):
    global last_command
    if direction != last_command:
        def target():
            try:
                requests.get(ROBOT_CMD_URL, params={'dir': direction}, timeout=0.2)
            except:
                pass 
        
        threading.Thread(target=target, daemon=True).start()
        last_command = direction
        print(f"🤖 Robot Command: {direction}")

# --- CAMERA HEARTBEAT THREAD (THE FIX) ---
def keep_alive_loop():
    while running:
        try:
            # Send handshake every 5 seconds to reset the 10s timer on ESP32
            requests.get(CAM_HANDSHAKE_URL, timeout=2)
            # print("💓 Keep-alive sent") 
        except:
            print("⚠ Keep-alive failed (Camera might be disconnected)")
        time.sleep(5) 

# --- MAIN FUNCTION ---
def main():
    global running

    # 1. Initial Handshake
    try:
        print("🔗 Connecting to Camera...")
        requests.get(CAM_HANDSHAKE_URL, timeout=3)
    except:
        print("❌ Could not connect to Camera. Check IP.")
        return

    # 2. Start the Keep-Alive Thread
    heartbeat_thread = threading.Thread(target=keep_alive_loop, daemon=True)
    heartbeat_thread.start()

    # 3. Open Video Stream
    cap = cv2.VideoCapture(CAM_STREAM_URL)
    time.sleep(1) # Give it a second to stabilize

    if not cap.isOpened():
        print("❌ Stream failed to open.")
        running = False
        return

    print("✅ System Online.")
    print("🎮 CONTROLS: Click Video Window -> WASD to Move, Q to Quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Stream lost (Check WiFi or Power)...")
            break

        # Run YOLO
        results = model(frame, verbose=False)
        
        # Draw Detections
        for obj in results[0].boxes:
            x1, y1, x2, y2 = map(int, obj.xyxy[0])
            conf = float(obj.conf[0])
            label = f"{model.names[int(obj.cls[0])]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Robot Vision", frame)

        # --- CONTROLS ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            send_robot_command('stop')
            running = False
            break
        elif key == ord('w'): send_robot_command('forward')
        elif key == ord('s'): send_robot_command('backward')
        elif key == ord('a'): send_robot_command('left')
        elif key == ord('d'): send_robot_command('right')
        elif key == 255:      send_robot_command('stop')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()