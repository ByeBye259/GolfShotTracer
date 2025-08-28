# GolfShotTracer: How it Works

## 🎯 Core Pipeline

### 1. Frame Preprocessing
- **Color Space Conversion**: Converts frames to LAB color space for better lighting invariance
- **Contrast Enhancement**: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Sharpening**: Uses custom kernel for edge enhancement
- **Background Subtraction**: MOG2 algorithm for motion detection
- **Noise Reduction**: Advanced morphological operations for clean segmentation

### 2. Ball Detection
- **Combined Detector**: Uses both YOLO and Roboflow models for robust detection
- **Contour Analysis**: Filters candidates by size, shape, and circularity
- **Motion Validation**: Validates detections with temporal consistency checks
- **Color Segmentation**: Identifies white/yellow golf balls in HSV color space

### 3. Tracking System
- **Kalman Filter**: Predicts ball position between frames
- **Hungarian Algorithm**: For data association between detections and tracks
- **Track Management**: Handles track initiation, update, and termination
- **Trajectory Smoothing**: Moving average filter for smooth visualization

### 4. Visualization
- **Real-time Overlay**: Shows tracking status and debug information
- **Trajectory Drawing**: Smooth path visualization with configurable styles
- **Performance Metrics**: Displays FPS and tracking statistics

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   Video Input   │───▶│  Preprocessing  │───▶│  Ball Detection │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│  Visualization  ◀────│  Trajectory     │◀───│  Multi-Object   │
│                 │    │  Tracking       │    │  Tracking       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🏆 Features

### Advanced Detection
- Combined YOLO + Roboflow models for high accuracy
- Adaptive thresholding for various lighting conditions
- Motion-based validation to reduce false positives

### Professional Tracking
- Kalman filter for smooth trajectory prediction
- Multi-hypothesis tracking for occlusions
- Automatic track management

### Performance Optimizations
- Efficient frame processing pipeline
- Configurable parameters for different hardware
- Real-time performance on consumer GPUs

## 📈 Performance Considerations

- **Resolution**: Works best at 720p or higher
- **Framerate**: Higher framerates (60fps+) improve tracking accuracy
- **Lighting**: Good lighting conditions significantly improve detection
- **Background**: Clean, static backgrounds yield best results

## 🚀 Future Improvements

- 3D trajectory reconstruction
- Ball speed and club head speed estimation
- Swing analysis metrics
- Mobile app integration
