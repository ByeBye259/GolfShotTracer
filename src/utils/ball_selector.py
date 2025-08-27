import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

class BallSelector:
    def __init__(self, window_name: str = "Select Ball Position"):
        self.window_name = window_name
        self.selected_pos: Optional[Tuple[int, int]] = None
        self.done = False
        self.frame = None
        self.display_frame = None
        self.zoom_scale = 1.0
        self.pan_x, self.pan_y = 0, 0
        self.is_panning = False
        self.last_x, self.last_y = 0, 0
        self.scale_x, self.scale_y = 1.0, 1.0
        
    def _mouse_callback(self, event, x: int, y: int, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coordinates to original image coordinates
            orig_x = int((x + self.pan_x) / self.zoom_scale * self.scale_x)
            orig_y = int((y + self.pan_y) / self.zoom_scale * self.scale_y)
            self.selected_pos = (orig_x, orig_y)
            self.done = True
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True  # Allow right-click to skip
            
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom in/out with mouse wheel
            zoom_factor = 1.1
            if flags > 0:  # Zoom in
                self.zoom_scale *= zoom_factor
            else:  # Zoom out
                self.zoom_scale /= zoom_factor
                if self.zoom_scale < 1.0:
                    self.zoom_scale = 1.0
                    self.pan_x, self.pan_y = 0, 0
            self._update_display()
            
        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle mouse button for panning
            self.is_panning = True
            self.last_x, self.last_y = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            # Pan the image
            dx, dy = x - self.last_x, y - self.last_y
            self.pan_x = max(0, min(self.pan_x - dx, self.display_frame.shape[1] * (self.zoom_scale - 1)))
            self.pan_y = max(0, min(self.pan_y - dy, self.display_frame.shape[0] * (self.zoom_scale - 1)))
            self.last_x, self.last_y = x, y
            self._update_display()
            
        elif event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False

    def _update_display(self):
        """Update the display with current zoom and pan settings."""
        if self.frame is None or self.display_frame is None:
            return
            
        # Apply zoom and pan
        h, w = self.display_frame.shape[:2]
        zoom_w = int(w / self.zoom_scale)
        zoom_h = int(h / self.zoom_scale)
        
        # Ensure pan values are within bounds
        self.pan_x = max(0, min(self.pan_x, w - zoom_w))
        self.pan_y = max(0, min(self.pan_y, h - zoom_h))
        
        # Create a view of the zoomed and panned region
        zoomed = self.display_frame[
            self.pan_y:self.pan_y + zoom_h,
            self.pan_x:self.pan_x + zoom_w
        ]
        
        # Resize back to window size
        zoomed = cv2.resize(zoomed, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Add instructions
        cv2.putText(zoomed, "Left-click: Select ball", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(zoomed, "Mouse wheel: Zoom in/out", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(zoomed, "Middle-click + drag: Pan", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(zoomed, "Right-click: Skip", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, zoomed)

    def _resize_to_fit_screen(self, frame, max_width: int = 1280, max_height: int = 800) -> np.ndarray:
        """Resize frame to fit screen while maintaining aspect ratio."""
        height, width = frame.shape[:2]
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Calculate new dimensions
        if width > max_width or height > max_height:
            if width / max_width > height / max_height:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            
            # Resize the frame
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return frame.copy()

    def select_ball(self, frame) -> Tuple[int, int]:
        """Show frame and wait for user to click on the ball position.
        
        Returns:
            Tuple[int, int]: (x, y) coordinates of selected position, or (-1, -1) if skipped
        """
        # Reset state
        self.selected_pos = None
        self.done = False
        self.zoom_scale = 1.0
        self.pan_x, self.pan_y = 0, 0
        self.is_panning = False
        
        # Create a resized copy for display
        self.frame = frame
        self.display_frame = self._resize_to_fit_screen(frame)
        
        # Calculate scale factors for coordinate conversion
        self.scale_x = frame.shape[1] / self.display_frame.shape[1]
        self.scale_y = frame.shape[0] / self.display_frame.shape[0]
        
        # Create window and set callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowTitle(self.window_name, "Select Golf Ball (Zoom with mouse wheel, Pan with middle button)")
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Initial display
        self._update_display()
        
        # Wait for selection
        while not self.done:
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC key
                self.selected_pos = (-1, -1)
                break
                
        cv2.destroyWindow(self.window_name)
        
        if self.selected_pos is None:
            return (-1, -1)
            
        return self.selected_pos
        
        instructions = "Click on the ball, then press any key or right-click to skip"
        cv2.putText(display_frame, instructions, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display_frame, instructions, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        while not self.done:
            cv2.imshow(self.window_name, display_frame)
            if cv2.waitKey(10) & 0xFF == 27:  # ESC to exit
                break
                
        # Scale the selected position back to original frame coordinates
        if self.selected_pos:
            x, y = self.selected_pos
            self.selected_pos = (int(x * scale_x), int(y * scale_y))
        
        cv2.destroyWindow(self.window_name)
        return self.selected_pos if self.selected_pos else (-1, -1)
