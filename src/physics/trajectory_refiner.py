import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
import cv2
import logging

logger = logging.getLogger(__name__)

@dataclass
class TrajectoryPoint:
    """Represents a point in the ball's trajectory with kinematic properties."""
    x: float                    # x-coordinate in pixels
    y: float                    # y-coordinate in pixels (increasing downward)
    t: float                    # Timestamp in seconds
    vx: float = 0.0             # x-velocity in pixels/second
    vy: float = 0.0             # y-velocity in pixels/second
    ax: float = 0.0             # x-acceleration in pixels/second²
    ay: float = 0.0             # y-acceleration in pixels/second²
    confidence: float = 1.0     # Detection confidence (0-1)
    is_predicted: bool = False  # Whether this point was predicted (not observed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'x': self.x,
            'y': self.y,
            't': self.t,
            'vx': self.vx,
            'vy': self.vy,
            'ax': self.ax,
            'ay': self.ay,
            'confidence': self.confidence,
            'is_predicted': self.is_predicted
        }
    
    @property
    def speed(self) -> float:
        """Return the speed in pixels/second."""
        return np.hypot(self.vx, self.vy)
    
    @property
    def acceleration(self) -> float:
        """Return the total acceleration magnitude in pixels/second²."""
        return np.hypot(self.ax, self.ay)

class TrajectoryRefiner:
    """Refines the ball trajectory using physics-based constraints and smoothing.
    
    This class provides methods to:
    - Add new trajectory points with kinematic properties
    - Smooth the trajectory while preserving important features
    - Predict future ball positions
    - Analyze ball flight characteristics
    - Handle missing or noisy detections
    """
    
    def __init__(self, 
                g: float = 9.81, 
                dt: float = 1/30.0,
                max_missing_frames: int = 3,
                min_velocity: float = 1.0,
                max_acceleration: float = 100.0):
        """Initialize the trajectory refiner.
        
        Args:
            g: Acceleration due to gravity in pixels/s² (positive downward).
            dt: Time step between frames in seconds.
            max_missing_frames: Maximum number of consecutive frames with missing detections.
            min_velocity: Minimum velocity threshold in pixels/second.
            max_acceleration: Maximum allowed acceleration in pixels/s².
        """
        self.g = abs(g)  # Ensure positive gravity
        self.dt = max(1e-6, abs(dt))  # Avoid division by zero
        self.max_missing_frames = max(1, max_missing_frames)
        self.min_velocity = max(0.1, min_velocity)
        self.max_acceleration = max(1.0, max_acceleration)
        
        # Trajectory data
        self.trajectory: List[TrajectoryPoint] = []
        self.smoothed_trajectory: List[TrajectoryPoint] = []
        self.missing_frames = 0
        
        # Cached values for performance
        self._last_valid_point: Optional[TrajectoryPoint] = None
        self._last_velocity = (0.0, 0.0)
        self._last_acceleration = (0.0, 0.0)
    
    def add_point(self, x: float, y: float, t: float, confidence: float = 1.0) -> bool:
        """Add a new point to the trajectory with kinematic calculations.
        
        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels (increasing downward).
            t: Timestamp in seconds.
            confidence: Detection confidence (0-1).
            
        Returns:
            bool: True if point was added successfully, False otherwise.
        """
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(t):
            logger.warning(f"Invalid point coordinates: x={x}, y={y}, t={t}")
            return False
            
        # Create new point
        point = TrajectoryPoint(x=x, y=y, t=t, confidence=confidence)
        
        # Calculate velocities and accelerations if we have previous points
        if self._last_valid_point is not None:
            dt = max(1e-6, t - self._last_valid_point.t)
            
            # Calculate velocities
            vx = (x - self._last_valid_point.x) / dt
            vy = (y - self._last_valid_point.y) / dt
            
            # Apply velocity constraints
            speed = np.hypot(vx, vy)
            if speed > 0:
                # Scale down if velocity is too high (likely a detection error)
                if speed > self.min_velocity * 10:  # Arbitrary threshold
                    scale = (self.min_velocity * 10) / speed
                    vx *= scale
                    vy *= scale
                    logger.debug(f"Reduced velocity from {speed:.1f} to {self.min_velocity * 10:.1f}")
            
            point.vx, point.vy = vx, vy
            
            # Calculate accelerations
            if len(self.trajectory) >= 2:
                ax = (vx - self._last_velocity[0]) / dt
                ay = (vy - self._last_velocity[1]) / dt + self.g  # Add gravity
                
                # Apply acceleration constraints
                accel = np.hypot(ax, ay)
                if accel > self.max_acceleration:
                    scale = self.max_acceleration / accel
                    ax *= scale
                    ay *= scale
                    logger.debug(f"Reduced acceleration from {accel:.1f} to {self.max_acceleration:.1f}")
                
                point.ax, point.ay = ax, ay
                self._last_acceleration = (ax, ay)
            
            self._last_velocity = (vx, vy)
        
        # Add point to trajectory
        self.trajectory.append(point)
        self._last_valid_point = point
        self.missing_frames = 0
        
        return True
    
    def add_missing_point(self) -> bool:
        """Add a predicted point when detection is missing.
        
        Returns:
            bool: True if point was added, False if max missing frames reached.
        """
        if not self.trajectory or self.missing_frames >= self.max_missing_frames:
            return False
            
        self.missing_frames += 1
        last = self.trajectory[-1]
        
        # Predict next position using constant acceleration model
        t = last.t + self.dt
        x = last.x + last.vx * self.dt + 0.5 * last.ax * self.dt**2
        y = last.y + last.vy * self.dt + 0.5 * (last.ay + self.g) * self.dt**2
        
        # Create predicted point with reduced confidence
        point = TrajectoryPoint(
            x=x, y=y, t=t,
            vx=last.vx + last.ax * self.dt,
            vy=last.vy + (last.ay + self.g) * self.dt,
            ax=last.ax,
            ay=last.ay,
            confidence=max(0.1, last.confidence * 0.7),  # Reduce confidence
            is_predicted=True
        )
        
        self.trajectory.append(point)
        self._last_valid_point = point
        return True
    
    def _smooth_1d(self, y: np.ndarray, window_size: int = 5, polyorder: int = 2) -> np.ndarray:
        """Apply Savitzky-Golay filtering with edge handling."""
        if len(y) < window_size:
            return y
            
        # Ensure window_size is odd and less than number of points
        window_size = min(window_size, len(y))
        if window_size % 2 == 0:
            window_size = max(3, window_size - 1)
            
        try:
            return savgol_filter(y, window_size, polyorder)
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning(f"Savitzky-Golay filtering failed: {e}")
            return y
    
    def _detect_outliers(self, values: np.ndarray, threshold: float = 2.0) -> np.ndarray:
        """Detect outliers using median absolute deviation."""
        if len(values) < 3:
            return np.zeros_like(values, dtype=bool)
            
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return np.zeros_like(values, dtype=bool)
            
        modified_z_scores = 0.6745 * (values - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    def refine(self, 
              window_size: int = 5, 
              polyorder: int = 2,
              smooth_velocity: bool = True,
              smooth_acceleration: bool = True) -> List[TrajectoryPoint]:
        """Refine the trajectory using physics-based constraints and smoothing.
        
        Args:
            window_size: Window size for Savitzky-Golay filter (must be odd).
            polyorder: Polynomial order for Savitzky-Golay filter.
            smooth_velocity: Whether to smooth velocity estimates.
            smooth_acceleration: Whether to smooth acceleration estimates.
            
        Returns:
            List of refined trajectory points.
        """
        if len(self.trajectory) < 3:
            return self.trajectory.copy()
            
        # Extract trajectory data
        t = np.array([p.t for p in self.trajectory])
        x = np.array([p.x for p in self.trajectory])
        y = np.array([p.y for p in self.trajectory])
        conf = np.array([p.confidence for p in self.trajectory])
        
        # 1. Remove duplicate timestamps
        _, unique_indices = np.unique(t, return_index=True)
        if len(unique_indices) < len(t):
            t = t[unique_indices]
            x = x[unique_indices]
            y = y[unique_indices]
            conf = conf[unique_indices]
            logger.debug(f"Removed {len(self.trajectory) - len(t)} duplicate timestamps")
        
        # 2. Interpolate missing values (if any)
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            valid = ~(np.isnan(x) | np.isnan(y))
            if np.sum(valid) >= 2:  # Need at least 2 points for interpolation
                x = np.interp(t, t[valid], x[valid])
                y = np.interp(t, t[valid], y[valid])
        
        # 3. Smooth positions
        x_smooth = self._smooth_1d(x, window_size, polyorder)
        y_smooth = self._smooth_1d(y, window_size, polyorder)
        
        # 4. Calculate velocities from smoothed positions
        dt = np.diff(t, prepend=t[0])
        vx = np.gradient(x_smooth, t)
        vy = np.gradient(y_smooth, t)
        
        # Smooth velocities if requested
        if smooth_velocity and len(t) >= window_size:
            vx = self._smooth_1d(vx, window_size, polyorder)
            vy = self._smooth_1d(vy, window_size, polyorder)
        
        # 5. Calculate accelerations
        ax = np.gradient(vx, t)
        ay = np.gradient(vy, t) + self.g  # Add gravity
        
        # Smooth accelerations if requested
        if smooth_acceleration and len(t) >= window_size:
            ax = self._smooth_1d(ax, window_size, polyorder)
            ay = self._smooth_1d(ay, window_size, polyorder)
        
        # 6. Create refined trajectory points
        refined = []
        for i in range(len(t)):
            point = TrajectoryPoint(
                x=float(x_smooth[i]),
                y=float(y_smooth[i]),
                t=float(t[i]),
                vx=float(vx[i]),
                vy=float(vy[i]),
                ax=float(ax[i]),
                ay=float(ay[i]),
                confidence=float(conf[i] if i < len(conf) else 1.0),
                is_predicted=getattr(self.trajectory[i], 'is_predicted', False) if i < len(self.trajectory) else False
            )
            refined.append(point)
        
        self.smoothed_trajectory = refined
        return refined
    
    def predict_future_positions(self, duration: float = 1.0, dt: Optional[float] = None) -> List[TrajectoryPoint]:
        """Predict future ball positions using current kinematic state.
        
        Args:
            duration: Duration to predict in seconds.
            dt: Time step between predicted points (defaults to instance dt).
            
        Returns:
            List of predicted trajectory points.
        """
        if not self.smoothed_trajectory:
            return []
            
        dt = dt or self.dt
        last = self.smoothed_trajectory[-1]
        
        # Use current state for prediction
        x0, y0 = last.x, last.y
        vx0, vy0 = last.vx, last.vy
        ax0, ay0 = last.ax, last.ay
        
        # Generate time points
        t = np.arange(0, duration + dt, dt)
        
        # Predict positions using constant acceleration model (with gravity)
        x = x0 + vx0 * t + 0.5 * ax0 * t**2
        y = y0 + vy0 * t + 0.5 * (ay0 + self.g) * t**2
        
        # Predict velocities
        vx = vx0 + ax0 * t
        vy = vy0 + (ay0 + self.g) * t
        
        # Create predicted points
        predicted = []
        for i in range(1, len(t)):  # Skip t=0 (current position)
            point = TrajectoryPoint(
                x=float(x[i]),
                y=float(y[i]),
                t=last.t + t[i],
                vx=float(vx[i]),
                vy=float(vy[i]),
                ax=float(ax0),
                ay=float(ay0),
                confidence=max(0.1, last.confidence * np.exp(-0.5 * i)),  # Decaying confidence
                is_predicted=True
            )
            predicted.append(point)
            
        return predicted
    
    def analyze_shot(self) -> Dict[str, Any]:
        """Analyze the golf shot and return key metrics.
        
        Returns:
            Dictionary containing shot analysis metrics.
        """
        if len(self.smoothed_trajectory) < 3:
            return {}
            
        traj = self.smoothed_trajectory
        
        # Basic metrics
        start = traj[0]
        apex = min(traj, key=lambda p: p.y)  # Highest point (minimum y)
        end = traj[-1]
        
        # Calculate carry distance (2D distance from start to end)
        carry_distance = np.hypot(end.x - start.x, end.y - start.y)
        
        # Calculate apex height (relative to start)
        apex_height = start.y - apex.y  # y increases downward
        
        # Calculate launch angle (degrees from horizontal)
        if len(traj) > 1:
            # Use first few points to estimate launch angle
            dx = traj[1].x - start.x
            dy = start.y - traj[1].y  # y increases downward
            launch_angle = np.degrees(np.arctan2(dy, dx))
        else:
            launch_angle = 0.0
        
        # Calculate ball speed at impact (if available)
        impact_speed = traj[-1].speed if hasattr(traj[-1], 'speed') else 0.0
        
        # Calculate hang time
        hang_time = end.t - start.t
        
        return {
            'carry_distance': float(carry_distance),
            'apex_height': float(apex_height),
            'launch_angle': float(launch_angle),
            'impact_speed': float(impact_speed),
            'hang_time': float(hang_time),
            'start_position': (float(start.x), float(start.y)),
            'end_position': (float(end.x), float(end.y)),
            'apex_position': (float(apex.x), float(apex.y)),
            'num_points': len(traj)
        }
    
    def reset(self):
        """Reset the trajectory refiner to its initial state."""
        self.trajectory = []
        self.smoothed_trajectory = []
        self.missing_frames = 0
        self._last_valid_point = None
        self._last_velocity = (0.0, 0.0)
        self._last_acceleration = (0.0, 0.0)
