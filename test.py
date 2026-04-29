import tkinter as tk
import requests

# --- CONFIGURATION (STATION MODE) ---
# In station mode the ESP32 gets an IP from your router (DHCP).
# Replace the default below with the IP shown in the ESP32 serial monitor or your router's client list.
ESP_IP = "192.168.137.65"   # <-- update this to your ESP32 station IP
URL = f"http://{ESP_IP}/command"

# --- LOGIC ---
last_command = "stop"

def set_target_ip(ip):
    global ESP_IP, URL
    ESP_IP = ip.strip()
    URL = f"http://{ESP_IP}/command"
    label_ip.config(text=f"Target: {ESP_IP}")

def send_command(direction):
    global last_command
    # Only send the request if the command has changed (prevents flooding)
    if direction != last_command:
        try:
            print(f"Sending: {direction} -> {URL}")
            requests.get(URL, params={'dir': direction}, timeout=0.5)
            last_command = direction
            update_label(direction)
        except requests.exceptions.RequestException:
            print("Error: Could not connect to Robot. Check network and IP.")
            update_label("error")

def on_key_press(event):
    key = event.keysym
    if key == 'Up':
        send_command('forward')
    elif key == 'Down':
        send_command('backward')
    elif key == 'Left':
        send_command('left')
    elif key == 'Right':
        send_command('right')

def on_key_release(event):
    # When you release a key, send stop
    if event.keysym in ['Up', 'Down', 'Left', 'Right']:
        send_command('stop')

def update_label(text):
    label_status.config(text=f"Status: {text.upper()}")

def apply_ip():
    ip = ip_entry.get()
    if ip:
        set_target_ip(ip)

# --- GUI SETUP ---
root = tk.Tk()
root.title("Robot Controller (Station Mode)")
root.geometry("360x220")

label_status = tk.Label(root, text="Connect WiFi & Press Arrows", font=("Arial", 14))
label_status.pack(pady=(10,5))

label_ip = tk.Label(root, text=f"Target: {ESP_IP}", font=("Arial", 10), fg="blue")
label_ip.pack()

frame = tk.Frame(root)
frame.pack(pady=8)

ip_entry = tk.Entry(frame, width=22)
ip_entry.insert(0, ESP_IP)
ip_entry.grid(row=0, column=0, padx=(0,6))

btn_apply = tk.Button(frame, text="Set IP", command=apply_ip)
btn_apply.grid(row=0, column=1)

instruction = tk.Label(root, text="Click this window to focus, use Arrow keys to control", font=("Arial", 10), fg="gray")
instruction.pack(pady=8)

# Bind keyboard events
root.bind('<KeyPress>', on_key_press)
root.bind('<KeyRelease>', on_key_release)

print("Controller started. Make sure your PC is on the same network as the ESP32 (station mode).")
root.mainloop()