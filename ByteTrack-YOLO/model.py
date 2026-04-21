"""
Script to copy YOLO model from training directory to project
"""

import shutil
from pathlib import Path

# Source: Your trained model
source_model = Path(r"D:\Learn\Year4\KLTN\Dataset\traffic_yolo_v11m_mixclass\best (8).pt")

# Destination: Project models directory
project_root = Path(__file__).parent
dest_model = project_root / "models" / "yolo11s_traffic.pt"

print("=" * 60)
print("Copy YOLO Model to Project")
print("=" * 60)
print(f"Source: {source_model}")
print(f"Destination: {dest_model}")
print()

if not source_model.exists():
    print(f" Error: Source model not found!")
    print(f"Please check the path: {source_model}")
    exit(1)

# Create models directory if not exists
dest_model.parent.mkdir(parents=True, exist_ok=True)

# Copy model
print("Copying model...")
shutil.copy2(source_model, dest_model)

# Check size
size_mb = dest_model.stat().st_size / (1024 * 1024)
print(f" Model copied successfully!")
print(f"  Size: {size_mb:.2f} MB")
print(f"  Location: {dest_model}")
print()
print("=" * 60)
print("You can now run the application:")
print("  python run_gui.py")
print("  or")
print("  python main.py --video your_video.mp4")
print("=" * 60)
