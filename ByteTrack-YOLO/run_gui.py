"""
ByteTrack-YOLO GUI Launcher
Launch the graphical user interface for video tracking
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.gui.app import create_gui


def main():
    """Main entry point for GUI"""
    print("=" * 60)
    print("ByteTrack-YOLO: Multi-Object Tracking System")
    print("GUI Mode")
    print("=" * 60)
    print()
    
    # Default model path
    default_model = Path(__file__).parent / "models" / "yolo11s_traffic.pt"
    
    if not default_model.exists():
        print(f"Warning: Default model not found at {default_model}")
        print("You can still use the GUI, but make sure to have a model file ready.")
        print()
        default_model = None
    else:
        default_model = str(default_model)
        print(f"Default model: {Path(default_model).name}")
        print()
    
    # Launch GUI
    print("Launching GUI...")
    create_gui(default_model)


if __name__ == '__main__':
    main()
