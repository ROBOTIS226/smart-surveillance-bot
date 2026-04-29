"""
object_segmentation.py – Device-adaptive segmentation wrapper
-------------------------------------------------------------
• CUDA / MPS / CPU auto-selection
• Supports Ultralytics YOLO-Seg *.pt or Torch-hub Mask-RCNN
• Mixed-precision on GPU, FP32 on CPU
• One-time warm-up
• infer() returns polygons or binary masks + metadata
"""


from __future__ import annotations
import cv2, numpy as np, time, torch
from typing import List, Dict, Optional
from camera import Camera


try:
    from ultralytics import YOLO          # YOLOv8-Seg
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False



class ObjectSegmentation:
    # ---------------- ctor ---------------- #
    def __init__(
        self,
        model_path: str | None = None,       # if None → use Torch-hub Mask-RCNN
        confidence: float = 0.5,
        use_gpu: bool = True,
        allow: Optional[List[str]] = None,
        block: Optional[List[str]] = None,
    ) -> None:


        self.conf = confidence
        self.allow = set(allow) if allow else None
        self.block = set(block) if block else None
        self.device = self._pick_device(use_gpu)
        print(f"[Seg] Device → {self.device}")


        # -------- load model ---------- #
        if model_path and YOLO_AVAILABLE:
            self.engine = "yoloseg"
            self.model = YOLO(model_path)
            if self.device == "cuda":
                self.model.half()
            self.model.to(self.device)
            self.names = self.model.names
        else:
            self.engine = "maskrcnn"
            self.model = torch.hub.load(
                "pytorch/vision", "maskrcnn_resnet50_fpn", pretrained=True
            )
            self.model.eval().to(self.device)
            self.names = [
                "__background__",
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "airplane",
                "bus",
                "train",
                "truck",
                "boat",
                "traffic light",
                "fire hydrant",
                "stop sign",
                "parking meter",
                "bench",
                "bird",
                "cat",
                "dog",
                "horse",
                "sheep",
                "cow",
                "elephant",
                "bear",
                "zebra",
                "giraffe",
                "backpack",
                "umbrella",
                "handbag",
                "tie",
                "suitcase",
                "frisbee",
                "skis",
                "snowboard",
                "sports ball",
                "kite",
                "baseball bat",
                "baseball glove",
                "skateboard",
                "surfboard",
                "tennis racket",
                "bottle",
                "wine glass",
                "cup",
                "fork",
                "knife",
                "spoon",
                "bowl",
                "banana",
                "apple",
                "sandwich",
                "orange",
                "broccoli",
                "carrot",
                "hot dog",
                "pizza",
                "donut",
                "cake",
                "chair",
                "couch",
                "potted plant",
                "bed",
                "dining table",
                "toilet",
                "tv",
                "laptop",
                "mouse",
                "remote",
                "keyboard",
                "cell phone",
                "microwave",
                "oven",
                "toaster",
                "sink",
                "refrigerator",
                "book",
                "clock",
                "vase",
                "scissors",
                "teddy bear",
                "hair dryer",
                "toothbrush",
            ]


        # warm-up
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self._forward(dummy)
        if self.device == "cuda":
            torch.cuda.synchronize()
        print("[Seg] Warm-up complete")


        # unique colors per class
        self.colors = np.random.uniform(0, 255, (len(self.names), 3))


    # ------------- helpers --------------- #
    @staticmethod
    def _pick_device(use_gpu: bool) -> str:
        if use_gpu and torch.cuda.is_available():
            return "cuda"
        if use_gpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


    def _forward(self, frame):
        if self.engine == "yoloseg":
            return self.model(frame, conf=self.conf, device=self.device, verbose=False)[0]
        else:                                 # maskrcnn: expects tensor CHW 0-1
            t = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
            with torch.no_grad():
                out = self.model([t.to(self.device)])[0]
            return out


    # ------------- public infer ---------- #
    def infer(self, frame) -> List[Dict]:
        """
        Returns list of:
        {cls_name, cls_id, conf, mask (H×W bool), bbox xyxy}
        """
        dets: List[Dict] = []
        h, w = frame.shape[:2]
        result = self._forward(frame)


        if self.engine == "yoloseg":
            for mask, box, conf, cls_id in zip(
                result.masks.data, result.boxes.xyxy, result.boxes.conf, result.boxes.cls
            ):
                conf = float(conf)
                cls_id = int(cls_id)
                name = self.names[cls_id]


                if self._skip(name, conf):
                    continue


                mask_np = mask.cpu().numpy() > 0.5
                x1, y1, x2, y2 = map(int, box)
                dets.append(
                    dict(
                        cls_name=name,
                        cls_id=cls_id,
                        conf=conf,
                        mask=mask_np,
                        bbox=(x1, y1, x2, y2),
                    )
                )


        else:  # maskrcnn
            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()
            boxes = result["boxes"].cpu().numpy()
            masks = result["masks"].cpu().numpy()[:, 0]  # NxH×W


            for score, cls_id, box, mask in zip(scores, labels, boxes, masks):
                conf = float(score)
                name = self.names[cls_id]


                if self._skip(name, conf):
                    continue


                mask_np = cv2.resize(mask, (w, h)) > 0.5
                x1, y1, x2, y2 = map(int, box)
                dets.append(
                    dict(
                        cls_name=name,
                        cls_id=int(cls_id),
                        conf=conf,
                        mask=mask_np,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return dets


    def _skip(self, name: str, conf: float) -> bool:
        if conf < self.conf:
            return True
        if self.allow and name not in self.allow:
            return True
        if self.block and name in self.block:
            return True
        return False


    # ------------- drawing --------------- #
    def draw(self, frame, dets: List[Dict]):
        overlay = frame.copy()
        for d in dets:
            color = self.colors[d["cls_id"]]
            mask = d["mask"]


            # colored mask overlay
            overlay[mask] = 0.6 * overlay[mask] + 0.4 * color


            # bbox + label
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d['cls_name']} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return cv2.addWeighted(frame, 1.0, overlay, 0.5, 0)


    # ------------- device info ------------- #
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


    # ------------- demo loop ------------- #
    def demo(self, cam_index: int = 0, fps_win: int = 30):
        cam = Camera(cam_index, fps=20)
        # Create window explicitly to avoid NULL guiReceiver error
        cv2.namedWindow("Segmentation", cv2.WINDOW_NORMAL)
        cam.start_capture()
        info = self.device_info()
        print(f"[Seg] Demo – press 'q' or close the window to quit  |  {info.get('name', self.device)}")

        ctr, t0 = 0, time.time()
        try:
            while True:
                frame, _ = cam.get_frame()
                if frame is None:
                    continue

                dets = self.infer(frame)
                vis = self.draw(frame, dets)

                # FPS meter
                ctr += 1
                if ctr % fps_win == 0:
                    fps = fps_win / (time.time() - t0)
                    t0 = time.time()
                else:
                    fps = None

                cv2.putText(
                    vis,
                    f"Seg: {len(dets)}  Conf>{self.conf}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                if fps:
                    cv2.putText(
                        vis,
                        f"FPS: {fps:.1f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                cv2.imshow("Segmentation", vis)
                key = cv2.waitKey(1) & 0xFF
                
                # Check for 'q' key
                if key == ord("q"):
                    break

                # Check if window was closed (but do it safely)
                try:
                    if cv2.getWindowProperty("Segmentation", cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    # Window was closed or destroyed
                    break

        except KeyboardInterrupt:
            print("[Seg] Interrupted by Ctrl+C")
        except Exception as e:
            print(f"[Seg] Error: {e}")
        finally:
            print("[Seg] Cleaning up...")
            cam.stop_capture()
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Flush GUI events
            print("[Seg] Demo ended")


if __name__ == "__main__":
    try:
        # A. YOLOv8-Seg weights (requires 'yolo11n-seg.pt')
        seg = ObjectSegmentation(model_path="yolo11n-seg.pt",
                                 confidence=0.5,
                                 use_gpu=True,
                                 allow=["person", "cat"])   # optional filter
        print(seg.device_info())
        seg.demo()
    except KeyboardInterrupt:
        print("\n[Main] Program interrupted")
    except Exception as e:
        print(f"[Main] Error: {e}")
    finally:
        print("[Main] Program terminated")
        import sys
        sys.exit(0)


# # B. Mask-RCNN fallback (no model_path given)
# seg_cpu = ObjectSegmentation(confidence=0.6, use_gpu=False)
# frame = cv2.imread("test.jpg")
# dets = seg_cpu.infer(frame)
# out  = seg_cpu.draw(frame, dets)
# cv2.imwrite("seg_result.png", out)
