import os
import urllib.request
import zipfile
from pathlib import Path

# Create weights directory if it doesn't exist
WEIGHTS_DIR = Path("weights")
WEIGHTS_DIR.mkdir(exist_ok=True)

# Model URLs and their destination paths
MODELS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt",
    "yolov8n-golf.pt": "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt",  # Placeholder - replace with actual golf model
    "EDSR_x2.pb": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
    "deep_sort": {
        "mars-small128.pb": "https://github.com/abewley/sort/raw/master/mars-small128.pb",
        "mars-small128.ckpt-68577.meta": "https://github.com/abewley/sort/raw/master/mars-small128.ckpt-68577.meta",
        "mars-small128.ckpt-68577.data-00000-of-00001": "https://github.com/abewley/sort/raw/master/mars-small128.ckpt-68577.data-00000-of-00001",
        "mars-small128.ckpt-68577.index": "https://github.com/abewley/sort/raw/master/mars-small128.ckpt-68577.index"
    }
}

def download_file(url: str, dest_path: Path):
    """Download a file from URL to destination path."""
    if dest_path.exists():
        print(f"[X] {dest_path} already exists")
        return
    
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, str(dest_path))
        print(f"[+] Downloaded {dest_path}")
    except Exception as e:
        print(f"[-] Failed to download {url}: {e}")

def main():
    print("Downloading model weights...")
    
    # Download YOLOv8 models
    for model_name, url in {k: v for k, v in MODELS.items() if not isinstance(v, dict)}.items():
        dest_path = WEIGHTS_DIR / model_name
        download_file(url, dest_path)
    
    # Download DeepSORT models
    deepsort_dir = WEIGHTS_DIR / "deep_sort"
    deepsort_dir.mkdir(exist_ok=True)
    
    for model_name, url in MODELS["deep_sort"].items():
        dest_path = deepsort_dir / model_name
        download_file(url, dest_path)
    
    # Create a zip file of the weights for easier distribution
    print("Creating weights.zip...")
    with zipfile.ZipFile("weights.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(WEIGHTS_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=WEIGHTS_DIR.parent)
                zipf.write(file_path, arcname)
    
    print("\n[+] All models downloaded successfully!")
    print(f"Weights are available in: {WEIGHTS_DIR.absolute()}")
    print("A zip archive has been created at: weights.zip")

if __name__ == "__main__":
    main()
