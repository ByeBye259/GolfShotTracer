# GolfShotTracer

Professional-grade golf shot tracer with advanced computer vision. Input a video (phone/action cam/DSLR), output the same video with an overlaid trajectory. Features advanced detection algorithms and smooth tracking. Runs locally from the terminal with optional preprocessing for enhanced accuracy.

## ✨ Features
- Advanced golf ball detection using combined YOLO and Roboflow models
- Professional-grade preprocessing (CLAHE, sharpening, contrast enhancement)
- Kalman filter-based trajectory prediction
- Real-time visualization with debug overlay
- Support for various video formats and resolutions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenCV 4.5+
- PyTorch (for YOLO model)
- FFmpeg (recommended for better video encoding)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/GolfShotTracer.git
cd GolfShotTracer
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/MacOS
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage
Process a golf shot video:
```bash
python scripts/process_video_combined.py --input path/to/your/video.mp4 --output output.mp4
```

### Advanced Options
```
--input INPUT         Path to input video file
--output OUTPUT       Path to save output video (default: output.mp4)
--config CONFIG       Path to config file (default: configs/detection_config.yaml)
--max_frames MAX      Maximum number of frames to process (for testing)
--no_preprocess       Disable frame preprocessing
--output_dir DIR      Directory to save output frames (optional)
```

## 📂 Project Structure
- `src/` - Core source code
  - `vision/` - Computer vision components
    - `pro_tracker.py` - Advanced golf ball tracker
    - `combined_detector.py` - Combined YOLO + Roboflow detector
- `configs/` - Configuration files
- `scripts/` - Utility scripts
  - `process_video_combined.py` - Main video processing script
  - `test_pro_tracker.py` - Tracker testing script
- `docs/` - Documentation
- `outputs/` - Default output directory

## 📚 Documentation
- [How It Works](docs/how-it-works.md) - Technical details about the implementation
- [Accuracy Notes](docs/accuracy.md) - Information about tracking accuracy
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## 🤝 Contributing
Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
