#!/usr/bin/env python3
"""
Test script for the enhanced TrajectoryRefiner class.
Simulates a golf ball trajectory and applies the refiner to smooth the data.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.physics.trajectory_refiner import TrajectoryRefiner

def simulate_golf_shot(
    launch_angle=30.0,  # degrees
    initial_speed=50.0,  # pixels/second
    duration=3.0,       # seconds
    dt=0.033,           # time step (30 FPS)
    noise_scale=2.0,    # scale of random noise to add
    initial_height=100.0  # initial height in pixels
):
    """Simulate a golf ball trajectory with optional noise.
    
    Args:
        launch_angle: Launch angle in degrees (0 = horizontal, 90 = vertical)
        initial_speed: Initial speed in pixels/second
        duration: Simulation duration in seconds
        dt: Time step in seconds
        noise_scale: Standard deviation of Gaussian noise to add
        initial_height: Initial height above ground in pixels
        
    Returns:
        Tuple of (time_array, x_positions, y_positions)
        Note: y=0 is the ground, positive y is upward
    """
    g = 9.81 * 5  # Scaled gravity for pixels (reduced for better visualization)
    
    # Convert launch angle to radians
    theta = np.radians(launch_angle)
    
    # Initial velocity components (y is positive upward)
    vx0 = initial_speed * np.cos(theta)
    vy0 = initial_speed * np.sin(theta)  # Positive because y increases upward
    
    # Time array
    t = np.arange(0, duration, dt)
    
    # Ideal trajectory (no air resistance)
    x = vx0 * t
    y = initial_height + vy0 * t - 0.5 * g * t**2
    
    # Add some noise to simulate detection errors
    x_noisy = x + np.random.normal(0, noise_scale, len(t))
    y_noisy = y + np.random.normal(0, noise_scale, len(t))
    
    # Remove points below ground level
    valid = y_noisy >= 0
    t = t[valid]
    x_noisy = x_noisy[valid]
    y_noisy = y_noisy[valid]
    
    return t, x_noisy, y_noisy

def main():
    print("Testing TrajectoryRefiner with simulated golf shot...")
    
    # Simulate a golf shot with more realistic parameters
    t, x, y = simulate_golf_shot(
        launch_angle=15.0,     # Lower angle for more realistic drive
        initial_speed=100.0,   # Higher initial speed
        duration=3.0,
        dt=0.033,             # ~30 FPS
        noise_scale=3.0,       # Slightly more noise
        initial_height=20.0    # Start slightly above ground
    )
    
    # Create and configure the trajectory refiner
    refiner = TrajectoryRefiner(
        g=9.81 * 5,   # Scaled gravity (must match simulation)
        dt=0.033,     # 30 FPS
        max_missing_frames=3,
        min_velocity=1.0,
        max_acceleration=100.0
    )
    
    # Add points to the refiner
    for i in range(len(t)):
        refiner.add_point(x[i], y[i], t[i], confidence=0.9)
        
        # Simulate occasional missing detections
        if i > 10 and i < 15 and i % 2 == 0:
            refiner.add_missing_point()
    
    # Refine the trajectory
    refined_traj = refiner.refine(
        window_size=7,
        polyorder=2,
        smooth_velocity=True,
        smooth_acceleration=True
    )
    
    # Analyze the shot
    analysis = refiner.analyze_shot()
    
    # Print analysis results
    print("\nShot Analysis:")
    print(f"- Carry distance: {analysis['carry_distance']:.1f} pixels")
    print(f"- Apex height: {analysis['apex_height']:.1f} pixels")
    print(f"- Launch angle: {analysis['launch_angle']:.1f}°")
    print(f"- Impact speed: {analysis['impact_speed']:.1f} px/s")
    print(f"- Hang time: {analysis['hang_time']:.2f} seconds")
    
    # Plot the results
    plt.figure(figsize=(14, 8))
    
    # Plot raw trajectory
    plt.plot(x, y, 'o', color='red', alpha=0.3, markersize=4, label='Raw Detection (Noisy)')
    
    # Plot refined trajectory
    x_refined = np.array([p.x for p in refined_traj])
    y_refined = np.array([p.y for p in refined_traj])
    
    # Plot the refined trajectory with a gradient to show confidence
    points = np.array([x_refined, y_refined]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a color gradient based on confidence
    confidences = np.array([p.confidence for p in refined_traj])
    norm = plt.Normalize(0.5, 1.0)
    cmap = plt.get_cmap('viridis')
    
    # Plot line segments with color gradient
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(confidences)
    line = plt.gca().add_collection(lc)
    plt.colorbar(line, label='Confidence')
    
    # Mark start and end points
    if refined_traj:
        plt.plot(refined_traj[0].x, refined_traj[0].y, 'go', markersize=10, label='Start')
        plt.plot(refined_traj[-1].x, refined_traj[-1].y, 'ro', markersize=10, label='End')
    
    # Add ground level
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add velocity vectors at key points
    num_vectors = min(5, len(refined_traj) // 3)
    for i in range(0, len(refined_traj), max(1, len(refined_traj) // num_vectors)):
        p = refined_traj[i]
        plt.arrow(p.x, p.y, p.vx/10, p.vy/10, head_width=5, head_length=5, fc='blue', ec='blue', alpha=0.7)
    
    # Add labels and title
    plt.title('Golf Ball Trajectory Refinement', fontsize=14, fontweight='bold')
    plt.xlabel('Horizontal Distance (pixels)', fontsize=12)
    plt.ylabel('Vertical Distance (pixels)', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add analysis text
    analysis_text = (
        f"Launch Angle: {analysis['launch_angle']:.1f}°\n"
        f"Carry Distance: {analysis['carry_distance']:.1f} px\n"
        f"Apex Height: {analysis['apex_height']:.1f} px\n"
        f"Hang Time: {analysis['hang_time']:.2f} s"
    )
    plt.text(0.02, 0.98, analysis_text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    output_dir = project_root / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'trajectory_refinement.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
