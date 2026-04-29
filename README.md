# Object Detection for Robotics 🤖

A high-performance, real-time object detection system designed for robotics applications using YOLO11n and optimized camera capture.

## 🚀 Features

- **Real-time Object Detection** with YOLO11n model
- **GPU Acceleration** (CUDA/MPS) with automatic CPU fallback
- **Low-latency Camera System** with threaded capture and queue management
- **80% Confidence Filtering** for high-quality detections
- **Thread-safe Architecture** optimized for robotics applications
- **Real-time Performance Monitoring** with FPS counter and device info

## 🏗️ Architecture

### Camera System (`camera.py`)
- **Thread-safe video capture** with automatic queue management
- **Anti-lag design**: Clears old frames to maintain real-time performance
- **Configurable FPS** with deterministic throttling (default: 30 FPS)
- **Memory-aware queue sizing** based on available system RAM

### Object Detection (`object_detection.py`)
- **Device-agnostic YOLO inference** with automatic hardware selection
- **Mixed-precision support**: FP16 on GPU, FP32 on CPU
- **80 COCO classes** pre-trained detection
- **Optional class filtering** (allow/block specific objects)
- **One-shot warm-up** to eliminate first-frame inference lag

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/saki7satvik/Object-Detection-for-robotics.git
   cd "Object Detection"
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # Linux/Mac
   # or
   myenv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install ultralytics opencv-python torch torchvision numpy psutil
   ```

4. **Download YOLO model** (if not already present):
   ```bash
   # The yolo11n.pt model should be in the project directory
   # It will be automatically downloaded on first run if missing
   ```

## 🎯 Usage

### Basic Object Detection Demo
```bash
python object_detection.py
```

### Programmatic Usage

#### Camera Only
```python
from camera import Camera

# Initialize camera
camera = Camera(camera_index=0, fps=30, max_queue_size=15)
camera.display_camera()  # Press 'q' to quit
```

#### Object Detection
```python
from object_detection import ObjectDetection

# Initialize detector
detector = ObjectDetection(
    model_path="yolo11n.pt",
    confidence=0.8,
    use_gpu=True
)

# Run live detection
detector.demo()
```

#### Custom Configuration
```python
# Detect only specific objects
detector = ObjectDetection(
    confidence=0.8,
    allow=["person", "car", "bicycle"],  # Only detect these classes
    use_gpu=True
)

# Get detection data programmatically
frame = cv2.imread("image.jpg")
detections = detector.infer(frame)

for obj in detections:
    print(f"Found {obj['cls_name']} with {obj['conf']:.2f} confidence")
    print(f"Location: {obj['center']}")
```

## 🔧 Configuration Options

### Camera Parameters
- `camera_index`: Camera device ID (default: 0)
- `fps`: Target frame rate (default: 30)
- `max_queue_size`: Frame buffer limit (auto-calculated based on RAM)

### Object Detection Parameters
- `model_path`: Path to YOLO model file (default: "yolo11n.pt")
- `confidence`: Detection confidence threshold (default: 0.8)
- `use_gpu`: Enable GPU acceleration (default: True)
- `allow`: List of allowed object classes (optional)
- `block`: List of blocked object classes (optional)

## 📊 Performance

### Hardware Requirements
- **Minimum**: CPU with 4GB RAM
- **Recommended**: NVIDIA GPU with 4GB VRAM or Apple Silicon Mac
- **Camera**: USB webcam or built-in camera

### Performance Benchmarks
- **CPU**: ~5-15 FPS (depends on processor)
- **NVIDIA GPU**: ~30-60 FPS (depends on GPU model)
- **Apple M1/M2**: ~20-40 FPS

## 🎮 Controls

### During Live Demo
- **'q' key**: Quit the application
- **ESC key**: Alternative quit method
- **Window close**: Click X to close

### Display Information
- **Green text**: Detection count and confidence threshold
- **Yellow text**: Current processing device
- **Cyan text**: Real-time FPS counter

## 🔍 Detected Objects

The system can detect 80 different object classes from the COCO dataset:

**People & Animals**: person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles**: bicycle, car, motorbike, aeroplane, bus, train, truck, boat

**Objects**: traffic light, fire hydrant, stop sign, parking meter, bench, backpack, umbrella, handbag, tie, suitcase, sports ball, kite, baseball bat, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, sofa, potted plant, bed, dining table, toilet, TV, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## 🛠️ Development

### Project Structure
```
Object Detection/
├── camera.py              # Thread-safe camera capture
├── object_detection.py    # YOLO-based object detection
├── object_segmentation.py # Object segmentation (future)
├── test.py               # Testing utilities
├── yolo11n.pt           # YOLO model weights
├── yolo11n-seg.pt       # Segmentation model weights
├── myenv/               # Virtual environment
└── README.md            # This file
```

### Key Classes

#### `Camera`
- `start_capture()`: Begin threaded frame capture
- `get_frame()`: Retrieve latest frame (non-blocking)
- `stop_capture()`: Clean shutdown with resource cleanup

#### `ObjectDetection`
- `infer(frame)`: Run detection on single frame
- `draw(frame, detections)`: Add bounding boxes and labels
- `demo()`: Live camera detection with display
- `device_info()`: Get current hardware information

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**Camera not working**:
- Check camera permissions
- Try different `camera_index` values (0, 1, 2...)
- Ensure camera is not being used by another application

**GPU not detected**:
- Install proper GPU drivers (NVIDIA CUDA or Apple Metal)
- Check PyTorch installation: `torch.cuda.is_available()`
- The system will automatically fall back to CPU

**Low FPS performance**:
- Close other applications using camera/GPU
- Reduce confidence threshold for fewer detections
- Use smaller input resolution if needed

**Model loading errors**:
- Ensure `yolo11n.pt` is in the project directory
- Check internet connection (model downloads automatically)
- Verify file permissions

