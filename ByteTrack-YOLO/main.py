#!/usr/bin/env python3
"""
ByteTrack-YOLO: Vehicle Tracking System with Traffic Violation Detection
Main entry point - launches GUI

Usage:
    python main.py

Features:
    - YOLO11 object detection
    - ByteTrack multi-object tracking
    - Traffic violation detection
    - Real-time video visualization
    - Interactive lane configuration (ROI drawing)
    - Video saving support
"""

if __name__ == '__main__':
    # Import after showing welcome message
    import tkinter as tk
    from tkinter import ttk
    from app_gui import ByteTrackGUI
    
    # Create and run GUI
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass  # Use default theme if clam not available
    
    app = ByteTrackGUI(root)
    
    print("=" * 60)
    print("ByteTrack GUI Started")
    print("=" * 60)
    
    root.mainloop()

