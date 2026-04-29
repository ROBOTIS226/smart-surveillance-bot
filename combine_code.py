"""
camera.py  –  Thread-safe, low-latency video capture for robotics
-----------------------------------------------------------------
• Deterministic FPS throttling
• Hardware time-stamps
• Self-clearing bounded queue
• Automatic camera re-open on failure
• Drop-frame metrics + hot-key viewer


Author: <your-name>
"""


from __future__ import annotations
import cv2
import threading, queue, time
from typing import Tuple, Optional, Dict
import psutil



class Camera:
    def __init__(
        self,
        camera_index: int = 0,
        fps: int = 30,
        max_queue_size: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        camera_index : int
            ID of /dev/video* (Linux) or DirectShow index (Win).
        fps : int
            Target capture rate; used for deterministic throttling.
        max_queue_size : int | None
            Upper bound on in-RAM frames.  If None, auto-computes from
            available system memory.
        """
        self.camera_index = camera_index
        self.fps = fps
        self.max_queue_size = (
            max_queue_size if max_queue_size is not None else self._auto_queue_size()
        )


        self.frame_queue: queue.Queue[dict] = queue.Queue()
        self.cap: Optional[cv2.VideoCapture] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.running = False


        # Metrics
        self.frames_captured = 0
        self.frames_delivered = 0


    # ---------------  private helpers  ---------------- #
    @staticmethod
    def _auto_queue_size() -> int:
        """Choose a queue size that keeps ≤256 MB total buffer."""
        free = psutil.virtual_memory().available
        bytes_per_frame = 640 * 480 * 3  # conservative default
        return max(1, min(20, free // (bytes_per_frame * 4)))


    def _open_camera(self) -> None:
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")


        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[Camera] Opened @ {self.cap.get(cv2.CAP_PROP_FPS):.1f} FPS")


    # ---------------  public initialiser  -------------- #
    def initialize_camera(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            self._open_camera()


    # ---------------  capture loop  -------------------- #
    def _capture_frames(self) -> None:
        interval = 1.0 / self.fps
        next_call = time.perf_counter()


        while self.running:
            if self.cap is None or not self.cap.isOpened():
                break
                
            ok, frame = self.cap.read()
            if not ok:
                print("[Camera] Read fail → reopen")
                time.sleep(0.5)
                try:
                    self._open_camera()
                    continue
                except RuntimeError:
                    print("[Camera] Failed to reopen camera")
                    break


            ts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1_000.0 or time.time()


            if self.frame_queue.qsize() >= self.max_queue_size:
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                print("[Camera] Queue overflow → flushed")


            try:
                self.frame_queue.put_nowait({"frame": frame, "timestamp": ts})
                self.frames_captured += 1
            except queue.Full:
                pass


            next_call += interval
            time.sleep(max(0.0, next_call - time.perf_counter()))


    # ---------------  public API  ---------------------- #
    def start_capture(self) -> None:
        if self.running:
            return
            
        self.initialize_camera()
        self.running = True
        self.capture_thread = threading.Thread(
            target=self._capture_frames, daemon=True
        )
        self.capture_thread.start()
        print("[Camera] Capture thread started")


    def get_frame(self) -> Tuple[Optional["cv2.Mat"], Optional[float]]:
        try:
            pkt = self.frame_queue.get_nowait()
            self.frames_delivered += 1
            return pkt["frame"].copy(), pkt["timestamp"]
        except queue.Empty:
            return None, None


    def get_stats(self) -> Dict[str, int]:
        return {
            "captured": self.frames_captured,
            "delivered": self.frames_delivered,
            "queue": self.frame_queue.qsize(),
        }


    def stop_capture(self) -> None:
        if not self.running:
            return
            
        print("[Camera] Stopping capture...")
        self.running = False
        
        # Wait for thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
            
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        print(f"[Camera] Stopped  |  {self.get_stats()}")


    def __del__(self):
        self.stop_capture()


    # ---------------  quick viewer  -------------------- #
    def display(self, window: str = "Camera") -> None:
        # Create window explicitly to avoid NULL guiReceiver error
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        
        if not self.running:
            self.start_capture()

        print("Keys: q quit | p pause | s save")
        paused = False
        frame, ts = None, None

        try:
            while True:
                if not paused:
                    frame, ts = self.get_frame()

                if frame is not None:
                    cv2.imshow(window, frame)

                key = cv2.waitKey(1) & 0xFF
                # Check for 'q' key or window closure
                if key == ord('q'):
                    break
                    
                # Check if window was closed (but do it safely)
                try:
                    if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    # Window was closed or destroyed
                    break
                    
                if key == ord('p'):
                    paused = not paused
                elif key == ord('s') and frame is not None:
                    name = f"frame_{int((ts or time.time())*1000)}.png"
                    cv2.imwrite(name, frame)
                    print(f"[Camera] Saved {name}")
                    
        except KeyboardInterrupt:
            print("[Camera] Interrupted by Ctrl+C")
        except Exception as e:
            print(f"[Camera] Error: {e}")
        finally:
            print("[Camera] Cleaning up...")
            self.stop_capture()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Flush GUI events
            print("[Camera] Display ended")



if __name__ == "__main__":
    try:
        Camera().display()
    except KeyboardInterrupt:
        print("\n[Main] Program interrupted")
    except Exception as e:
        print(f"[Main] Error: {e}")
    finally:
        print("[Main] Program terminated")
        import sys
        sys.exit(0)
