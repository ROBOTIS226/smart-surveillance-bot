"""
object_detection.py  –  Device-agnostic YOLO inference wrapper
--------------------------------------------------------------
• CUDA → MPS → CPU auto-selection
• Mixed-precision on GPU, FP32 on CPU
• One-shot warm-up to eliminate first-frame lag
• Optional class allow/block filters
• Headless infer() + optional draw() helpers
"""


from __future__ import annotations
import cv2, numpy as np, time
from typing import List, Dict, Optional
import torch
from ultralytics import YOLO
from camera import Camera



class ObjectDetection:
    # -------------------  ctor  -------------------- #
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.8,
        use_gpu: bool = True,
        allow: Optional[List[str]] = None,
        block: Optional[List[str]] = None,
    ) -> None:
        self.conf = confidence
        self.allow = set(allow) if allow else None
        self.block = set(block) if block else None
        self.device = self._pick_device(use_gpu)
        print(f"[OD] Device → {self.device}")


        # Load the model
        self.model = YOLO(model_path)


        #Move to device before fusing
        self.model.to(self.device)


        if self.device == "cuda":
            self.model.half()
        
        #Fuse the model for faster inference
        try:
            self.model.fuse()
        except Exception as e:
            print(f"[OD] Model fusion failed: {e}")


        self.names = self.model.names
        self.colors = np.random.uniform(0, 255, (len(self.names), 3))


        # Warm-up
        _ = self.model(
            np.zeros((640, 640, 3), dtype=np.uint8),
            device=self.device,
            verbose=False,
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        print("[OD] Warm-up complete")


    # -------------------  helpers  ----------------- #
    @staticmethod
    def _pick_device(use_gpu: bool) -> str:
        if use_gpu and torch.cuda.is_available():
            return "cuda"
        if use_gpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


    # -------------------  inference  --------------- #
    @torch.no_grad()
    def infer(self, frame) -> List[Dict]:
        res = self.model(frame, conf=self.conf, device=self.device, verbose=False)[0]
        dets: List[Dict] = []


        if res.boxes is None:
            return dets


        for b in res.boxes:
            conf = float(b.conf[0])
            cls_id = int(b.cls)
            name = self.names[cls_id]


            if self.allow and name not in self.allow:
                continue
            if self.block and name in self.block:
                continue
            if conf < self.conf:
                continue


            x1, y1, x2, y2 = map(int, b.xyxy[0])
            dets.append(
                dict(
                    cls_name=name,
                    cls_id=cls_id,
                    conf=conf,
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                )
            )
        return dets


    # -------------------  drawing  ----------------- #
    def draw(self, frame, detections: List[Dict]):
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cls_id, conf = d["cls_id"], d["conf"]
            color = self.colors[cls_id]


            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d['cls_name']} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame


    # -------------------  device info -------------- #
    def device_info(self):
        info = {"device": self.device}
        if self.device == "cuda":
            idx = torch.cuda.current_device()
            prop = torch.cuda.get_device_properties(idx)
            info |= dict(
                name=prop.name,
                mem_total_GB=prop.total_memory / 1e9,
                mem_alloc_GB=torch.cuda.memory_allocated(idx) / 1e9,
            )
        elif self.device == "mps":
            info["name"] = "Apple MPS"
        else:
            info["name"] = "CPU"
        return info


    # -------------------  demo loop  ---------------- #
    def demo(self, cam_index: int = 0, fps_win: int = 30):
        cam = Camera(cam_index, fps=30)
        # Create window explicitly to avoid NULL guiReceiver error
        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
        cam.start_capture()
        info = self.device_info()
        print(f"[OD] Demo – press 'q' or close the window to quit  |  {info['name']}")

        ctr, t0 = 0, time.time()
        try:
            while True:
                frame_data = cam.get_frame()
                if frame_data is None:
                    continue
                
                # Handle both tuple and dict return formats
                if isinstance(frame_data, tuple):
                    frame, _ = frame_data
                elif isinstance(frame_data, dict):
                    frame = frame_data.get("frame")
                else:
                    frame = frame_data
                
                if frame is None:
                    continue

                dets = self.infer(frame)
                vis = self.draw(frame, dets)

                # FPS overlay
                ctr += 1
                if ctr % fps_win == 0:
                    fps = fps_win / (time.time() - t0)
                    t0 = time.time()
                else:
                    fps = None

                cv2.putText(
                    vis,
                    f"Det: {len(dets)}  Conf>{self.conf}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    vis,
                    f"Device: {info['name'][:20]}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )
                if fps:
                    cv2.putText(
                        vis,
                        f"FPS: {fps:.1f}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                cv2.imshow("YOLO Detection", vis)
                key = cv2.waitKey(1) & 0xFF
                
                # Check for 'q' key
                if key == ord("q"):
                    break
                    
                # Check if window was closed (but do it safely)
                try:
                    if cv2.getWindowProperty("YOLO Detection", cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    # Window was closed or destroyed
                    break
                    
        except KeyboardInterrupt:
            print("[OD] Interrupted by Ctrl+C")
        except Exception as e:
            print(f"[OD] Error: {e}")
        finally:
            print("[OD] Cleaning up...")
            cam.stop_capture()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Flush GUI events
            print("[OD] Demo ended")



if __name__ == "__main__":
    try:
        od = ObjectDetection(model_path="yolo11n.pt", confidence=0.8, use_gpu=True)
        print(od.device_info())
        od.demo()
    except KeyboardInterrupt:
        print("\n[Main] Program interrupted")
    except Exception as e:
        print(f"[Main] Error: {e}")
    finally:
        print("[Main] Program terminated")
        import sys
        sys.exit(0)
