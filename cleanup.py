import os
import shutil

def cleanup_project():
    """Remove unnecessary files and keep only the essential ones."""
    # Files to keep
    essential_files = {
        'golf_tracer.py',
        'cleanup.py',
        'requirements.txt',
        'README.md',
        '.gitignore'
    }
    
    # Directories to keep
    essential_dirs = {
        'models',
        'outputs',
        'test_videos'
    }
    
    # Create essential directories if they don't exist
    for dir_name in essential_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # List all files in the current directory
    for item in os.listdir('.'):
        if os.path.isfile(item) and item not in essential_files and not item.startswith('.'):
            try:
                os.remove(item)
                print(f"Removed file: {item}")
            except Exception as e:
                print(f"Error removing {item}: {e}")
    
    # List all directories in the current directory
    for item in os.listdir('.'):
        if os.path.isdir(item) and item not in essential_dirs and not item.startswith('.'):
            try:
                shutil.rmtree(item)
                print(f"Removed directory: {item}")
            except Exception as e:
                print(f"Error removing directory {item}: {e}")
    
    # Clean up scripts directory if it exists
    if os.path.exists('scripts'):
        # Create scripts directory if it doesn't exist
        os.makedirs('scripts', exist_ok=True)
        
        # List all files in scripts directory
        for item in os.listdir('scripts'):
            item_path = os.path.join('scripts', item)
            if os.path.isfile(item_path) and item != 'cv_ball_detector.py':
                try:
                    os.remove(item_path)
                    print(f"Removed file: scripts/{item}")
                except Exception as e:
                    print(f"Error removing scripts/{item}: {e}")
    
    print("\nCleanup complete! The project now contains only essential files.")
    print("\nTo use the golf ball tracer, run:")
    print("python golf_tracer.py path/to/your/video.mp4 -o outputs/output.mp4")

if __name__ == "__main__":
    cleanup_project()
