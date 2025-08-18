import cv2
from typing import Optional, Tuple

class BallSelector:
    def __init__(self, window_name: str = "Select Ball Position"):
        self.window_name = window_name
        self.selected_pos: Optional[Tuple[int, int]] = None
        self.done = False

    def _mouse_callback(self, event, x: int, y: int, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_pos = (x, y)
            self.done = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.done = True  # Allow right-click to skip

    def _resize_to_fit_screen(self, frame, max_width: int = 1280, max_height: int = 800) -> tuple:
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
        return frame

    def select_ball(self, frame) -> Tuple[int, int]:
        """Show frame and wait for user to click on the ball position.
        
        Returns:
            Tuple[int, int]: (x, y) coordinates of selected position, or (-1, -1) if skipped
        """
        # Create a resized copy for display
        display_frame = self._resize_to_fit_screen(frame)
        scale_x = frame.shape[1] / display_frame.shape[1]
        scale_y = frame.shape[0] / display_frame.shape[0]
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowTitle(self.window_name, "Click the golf ball, then press any key")
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
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
