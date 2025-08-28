# 🛠️ Troubleshooting Guide

## 🎥 Video Input Issues

### Video Won't Load
- **Symptom**: Error loading video file
- **Solutions**:
  - Verify file path is correct
  - Check file permissions
  - Convert to a standard format (MP4 with H.264 codec recommended)
  - Ensure file is not corrupted

### Poor Video Quality
- **Symptom**: Blurry or pixelated video
- **Solutions**:
  - Use higher resolution (1080p or 4K)
  - Ensure proper lighting
  - Use a tripod to reduce camera shake
  - Disable digital zoom

## 🎯 Detection Issues

### Ball Not Detected
- **Symptom**: No ball detection in output
- **Solutions**:
  1. **Check Ball Visibility**
     - Ensure ball is in focus
     - Use high-visibility golf balls (white or yellow)
     - Avoid direct sunlight on the ball
  2. **Adjust Detection Settings**
     - Lower confidence threshold in config
     - Enable preprocessing
     - Adjust color thresholds for your environment
  3. **Camera Settings**
     - Increase shutter speed to reduce motion blur
     - Use manual focus
     - Ensure proper exposure

### False Detections
- **Symptom**: Incorrect objects being tracked
- **Solutions**:
  - Increase detection confidence threshold
  - Enable motion validation
  - Adjust ROI to exclude distracting elements
  - Use a clean background

## 🚀 Performance Issues

### Low Frame Rate
- **Symptom**: Processing is very slow
- **Solutions**:
  - Reduce resolution (try 720p)
  - Disable preprocessing if not needed
  - Use GPU acceleration if available
  - Process shorter video segments

### High CPU/GPU Usage
- **Symptom**: System becomes unresponsive
- **Solutions**:
  - Lower processing resolution
  - Close other applications
  - Reduce batch size in config
  - Enable hardware acceleration

## 📹 Output Issues

### No Output Video
- **Symptom**: Processing completes but no output file
- **Solutions**:
  - Check write permissions
  - Verify output directory exists
  - Ensure enough disk space
  - Try a different output format

### Poor Quality Output
- **Symptom**: Output video is low quality
- **Solutions**:
  - Increase output bitrate
  - Use lossless codec for intermediate files
  - Ensure proper resolution settings
  - Check compression settings

## 🔧 Common Error Messages

### "FFmpeg not found"
- **Solution**:
  ```bash
  # Windows (with Chocolatey)
  choco install ffmpeg
  
  # Ubuntu/Debian
  sudo apt install ffmpeg
  
  # macOS (with Homebrew)
  brew install ffmpeg
  ```

### "CUDA out of memory"
- **Solutions**:
  - Reduce batch size
  - Lower resolution
  - Close other GPU applications
  - Enable memory growth in config

### "Module not found"
- **Solution**:
  ```bash
  # Ensure all dependencies are installed
  pip install -r requirements.txt
  ```

## 🎚️ Advanced Troubleshooting

### Debug Mode
Enable debug logging for more detailed output:
```bash
python scripts/process_video_combined.py --input input.mp4 --output output.mp4 --debug
```

### Generate Debug Frames
Save intermediate processing frames:
```bash
python scripts/process_video_combined.py --input input.mp4 --output output.mp4 --debug_frames
```

### Performance Profiling
Profile the application to identify bottlenecks:
```bash
python -m cProfile -o profile.cprof scripts/process_video_combined.py --input input.mp4
```

## 📞 Getting Help

If you've tried these solutions and are still experiencing issues:
1. Check the [GitHub Issues](https://github.com/yourusername/GolfShotTracer/issues) for similar problems
2. Include the following in your bug report:
   - Your system specifications
   - Exact command used
   - Error messages
   - Sample video (if possible)
3. Create a new issue with the `bug` label

## 🔄 Known Issues

- **Memory Leaks**: Long videos may cause memory issues - process in segments
- **GPU Compatibility**: Some AMD GPUs may have compatibility issues
- **Windows Paths**: Use raw strings or double backslashes in Windows paths
- **Special Characters**: Avoid special characters in file paths

## 🛠️ Maintenance Tips

- Keep your dependencies updated:
  ```bash
  pip install --upgrade -r requirements.txt
  ```
- Clear cache files regularly
- Monitor system resources during processing
- Keep backups of your configuration files
