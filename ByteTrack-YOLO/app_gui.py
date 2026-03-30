#!/usr/bin/env python3
"""
ByteTrack GUI - Vehicle Tracking with YOLO11 and Traffic Violation Detection
GUI matching bytetrack_test.py interface but using modular src packages

Usage:
    python app_gui.py
"""

import sys
import os
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:
    messagebox.showerror("Error", "PIL not installed. Run: pip install pillow")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.detector import YOLODetector
from src.tracker import BYTETracker, STrack, TrackState, ViolationDetector, TrafficLane, ViolationType, TrackViolation


# ============================================================================
# ROI Selector - Interactive polygon drawing
# ============================================================================

class ROISelector:
    """Interactive ROI polygon selector - exactly 4 points for lane"""
    
    def __init__(self, frame, canvas, callback):
        """
        Args:
            frame: Original video frame (BGR)
            canvas: tkinter Canvas to draw on
            callback: Function to call when 4 points are drawn
        """
        self.frame = frame
        self.canvas = canvas
        self.callback = callback
        self.points = []
        
        # Bind events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        canvas.focus_set()
        
        self._update_display()
    
    def _on_canvas_click(self, event):
        """Handle canvas click"""
        # Map canvas coordinates back to frame coordinates
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        frame_h, frame_w = self.frame.shape[:2]
        
        scale = min(canvas_w / frame_w, canvas_h / frame_h)
        
        # Calculate offset (frame is centered on canvas)
        offset_x = (canvas_w - frame_w * scale) / 2
        offset_y = (canvas_h - frame_h * scale) / 2
        
        # Map click to frame coordinates
        frame_x = int((event.x - offset_x) / scale)
        frame_y = int((event.y - offset_y) / scale)
        
        # Clamp to frame bounds
        frame_x = max(0, min(frame_x, frame_w - 1))
        frame_y = max(0, min(frame_y, frame_h - 1))
        
        self.points.append((frame_x, frame_y))
        print(f"Point {len(self.points)}: ({frame_x}, {frame_y})")
        
        self._update_display()
        
        # If 4 points drawn, finish
        if len(self.points) >= 4:
            self.canvas.unbind("<Button-1>")
            self._finish()
    
    def _update_display(self):
        """Update canvas display with current points"""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            return
        
        # Resize frame to fit canvas
        frame_h, frame_w = self.frame.shape[:2]
        scale = min(canvas_w / frame_w, canvas_h / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        
        frame_resized = cv2.resize(self.frame, (new_w, new_h))
        
        # Draw points
        for i, (px, py) in enumerate(self.points):
            px_scaled = int(px * scale)
            py_scaled = int(py * scale)
            cv2.circle(frame_resized, (px_scaled, py_scaled), 5, (0, 255, 0), -1)
            cv2.putText(frame_resized, str(i+1), (px_scaled+10, py_scaled-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw polygon if have points
        if len(self.points) > 1:
            pts = np.array([(int(p[0]*scale), int(p[1]*scale)) for p in self.points], dtype=np.int32)
            cv2.polylines(frame_resized, [pts], False, (0, 255, 255), 2)
        
        # Convert and display
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=img)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor=tk.CENTER)
        self.canvas.image = photo
    
    def _finish(self):
        """Called when 4 points are drawn"""
        print(f"Polygon finished: {self.points}")
        self.callback(self.points)


# ============================================================================
# GUI Main Class
# ============================================================================

class ByteTrackGUI:
    """GUI Application for ByteTrack + YOLO video tracking"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("ByteTrack Tracker")
        self.root.geometry("1800x900")
        
        # State variables
        self.video_path = None
        self.output_path = None
        self.is_processing = False
        self.should_stop = False
        self.cap = None
        self.tracker = None
        self.detector = None
        
        # Traffic lane & violation
        self.violation_detector = ViolationDetector()
        self.traffic_lanes = {}
        self.selected_roi = None
        self.roi_selector_active = False
        
        # Queue for frame display
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Model path (hardcoded by default)
        self.model_path = r"D:\Learn\Year4\KLTN\Dataset\traffic_yolo_v11m_4class\best_(4).pt"
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets - LEFT settings panel + RIGHT video panel"""
        
        # Configure grid - settings on left (fixed), video on right (expandable)
        self.root.columnconfigure(0, weight=0, minsize=200)  # Left - fixed width
        self.root.columnconfigure(1, weight=1)  # Right - expandable
        self.root.rowconfigure(0, weight=1)
        
        # ===== LEFT PANEL: Settings =====
        left_frame = ttk.LabelFrame(self.root, text="⚙ Settings", padding="8")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), 
                       padx=3, pady=3, ipadx=5, ipady=5)
        left_frame.columnconfigure(0, weight=1)
        
        row = 0
        
        # --- Video Selection ---
        ttk.Label(left_frame, text="Video:", font=("Arial", 8, "bold")).grid(
            row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        self.video_label = ttk.Label(left_frame, text="None", 
                                     foreground="gray", font=("Arial", 7), wraplength=140)
        self.video_label.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 3))
        row += 1
        
        ttk.Button(left_frame, text="Browse", command=self.browse_video, width=15).grid(
            row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        row += 1
        
        # --- Model Path ---
        ttk.Label(left_frame, text="Model:", font=("Arial", 8, "bold")).grid(
            row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        model_label = ttk.Label(left_frame, text="YOLO11s", 
                               foreground="blue", font=("Arial", 7))
        model_label.grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        
        # --- Device Selection ---
        ttk.Label(left_frame, text="Device:", font=("Arial", 8, "bold")).grid(
            row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        self.device_var = tk.StringVar(value="cuda")
        device_frame = ttk.Frame(left_frame)
        device_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Radiobutton(device_frame, text="GPU", variable=self.device_var, 
                       value="cuda").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_var, 
                       value="cpu").pack(side=tk.LEFT)
        row += 1
        
        # --- Separator ---
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        row += 1
        
        # --- Detection Confidence ---
        ttk.Label(left_frame, text="Detection:", font=("Arial", 8, "bold")).grid(
            row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        self.det_conf_var = tk.DoubleVar(value=0.01)
        det_conf_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, 
                                   variable=self.det_conf_var, orient=tk.HORIZONTAL)
        det_conf_scale.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 1))
        row += 1
        
        self.det_conf_label = ttk.Label(left_frame, text="0.01", font=("Arial", 7))
        self.det_conf_label.grid(row=row, column=0, sticky=tk.W)
        det_conf_scale.config(command=lambda v: self.det_conf_label.config(
            text=f"{float(v):.2f}"))
        row += 1
        
        # --- Separator ---
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        row += 1
        
        # --- Track Settings ---
        ttk.Label(left_frame, text="Track Buf:", font=("Arial", 7)).grid(
            row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        self.track_buffer_var = tk.IntVar(value=30)
        buffer_spinbox = ttk.Spinbox(left_frame, from_=1, to=100, 
                                     textvariable=self.track_buffer_var, width=12)
        buffer_spinbox.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=1)
        row += 1
        
        ttk.Label(left_frame, text="Min Area:", font=("Arial", 7)).grid(
            row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        self.min_box_area_var = tk.IntVar(value=400)
        area_spinbox = ttk.Spinbox(left_frame, from_=100, to=5000, increment=100,
                                   textvariable=self.min_box_area_var, width=12)
        area_spinbox.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=1)
        row += 1
        
        ttk.Label(left_frame, text="Edge:", font=("Arial", 7)).grid(
            row=row, column=0, sticky=tk.W, pady=2)
        row += 1
        
        self.edge_margin_var = tk.IntVar(value=10)
        margin_spinbox = ttk.Spinbox(left_frame, from_=0, to=100, increment=5,
                                     textvariable=self.edge_margin_var, width=12)
        margin_spinbox.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=1)
        row += 1
        
        # --- Output Settings ---
        self.save_output_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Save output", 
                       variable=self.save_output_var,
                       command=self.toggle_output).grid(row=row, column=0, 
                                                        sticky=tk.W, pady=5)
        row += 1
        
        # --- Separator ---
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        row += 1
        
        # --- Violation Detection ---
        ttk.Label(left_frame, text="Violation Detection", 
                 font=("Arial", 8, "bold")).grid(row=row, column=0, sticky=tk.W, pady=3)
        row += 1
        
        self.enable_violation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Enable violation check", 
                       variable=self.enable_violation_var).grid(
                       row=row, column=0, sticky=tk.W, pady=1)
        row += 1
        
        # --- ROI Selection Buttons ---
        lane_btn_frame = ttk.Frame(left_frame)
        lane_btn_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        ttk.Button(lane_btn_frame, text="New Lane", 
                  command=self.setup_traffic_lanes, width=11).pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        ttk.Button(lane_btn_frame, text="Modify", 
                  command=self.modify_traffic_lane, width=8).pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)
        ttk.Button(lane_btn_frame, text="Delete", 
                  command=self.delete_traffic_lane, width=8).pack(side=tk.LEFT, padx=(1, 0), fill=tk.X, expand=True)
        row += 1
        
        # --- Lane Info ---
        self.lane_info_label = ttk.Label(left_frame, 
                                        text="No lanes configured", 
                                        foreground="orange", 
                                        font=("Arial", 7), 
                                        wraplength=140)
        self.lane_info_label.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        row += 1
        
        # --- Separator ---
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky=(tk.W, tk.E), pady=3)
        row += 1
        
        # --- Control Buttons ---
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.start_button = ttk.Button(button_frame, text="Start", 
                                       command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.stop_button = ttk.Button(button_frame, text="Stop", 
                                      command=self.stop_processing, 
                                      state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        row += 1
        
        # --- Spacer to push to top ---
        ttk.Label(left_frame, text="").grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10)
        left_frame.rowconfigure(row, weight=1)
        
        # ===== RIGHT PANEL: Video Display =====
        right_frame = ttk.Frame(self.root, padding="5")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=3, pady=3)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=0)
        
        # --- Large Video Canvas ---
        self.canvas = tk.Canvas(right_frame, width=1300, height=750, bg="black")
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        
        # --- Status & Progress (at bottom) ---
        status_frame = ttk.Frame(right_frame)
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready", font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate', length=300)
        self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
    
    def browse_video(self):
        """Open file dialog to select video"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.video_path = filename
            self.video_label.config(text=Path(filename).name, foreground="black")
            print(f"✓ Video selected: {filename}")
            
            # Clear lanes when video changes
            self.traffic_lanes.clear()
            self.violation_detector.lanes.clear()
            self.violation_detector.persistent_violations.clear()
            self.violation_detector.previous_positions.clear()
            self.update_lane_info_display()
            
            # Suggest output path
            if self.save_output_var.get():
                output_name = Path(filename).stem + "_tracked.mp4"
                self.output_path = str(Path(filename).parent / output_name)
    
    def toggle_output(self):
        """Toggle output video saving"""
        if self.save_output_var.get() and self.video_path:
            output_name = Path(self.video_path).stem + "_tracked.mp4"
            self.output_path = str(Path(self.video_path).parent / output_name)
        else:
            self.output_path = None
    
    def setup_traffic_lanes(self):
        """Setup traffic lanes - draw 4 points then configure vehicle types"""
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video first!")
            return
        
        # Open video to get first frame
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            messagebox.showerror("Error", "Cannot read video")
            return
        
        # Load detector to get class names
        try:
            detector_temp = YOLODetector(
                model_path=self.model_path,
                conf_threshold=self.det_conf_var.get(),
                device=self.device_var.get()
            )
            class_names = detector_temp.class_names
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load model: {str(e)}")
            return
        
        # Show canvas
        self.root.deiconify()
        self.canvas.focus_set()
        
        polygon_result = [None]
        
        def on_polygon_finished(polygon):
            """Called when 4 points are drawn"""
            polygon_result[0] = polygon
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Create dialog for vehicle type selection
            config_window = tk.Toplevel(self.root)
            config_window.title("Select Allowed Vehicle Types")
            config_window.geometry("400x400")
            config_window.resizable(False, False)
            
            ttk.Label(config_window, text=f"Lane {len(self.traffic_lanes) + 1}:", 
                     font=("Arial", 10, "bold")).pack(pady=10)
            ttk.Label(config_window, text="Which vehicle types are allowed in this lane?", 
                     font=("Arial", 9)).pack(pady=5)
            
            # Checkboxes for vehicle types
            class_vars = {}
            ttk.Label(config_window, text="Allowed vehicle types:", 
                     font=("Arial", 9, "bold")).pack(pady=(5, 10))
            cb_frame = ttk.Frame(config_window)
            cb_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
            
            for class_id, class_name in sorted(class_names.items()):
                var = tk.BooleanVar(value=False)
                # Default: select common vehicle types
                if 'car' in class_name.lower() or 'motorcycle' in class_name.lower():
                    var.set(True)
                
                cb = ttk.Checkbutton(cb_frame, text=f"{class_name} (ID:{class_id})", variable=var)
                cb.pack(anchor=tk.W, pady=3)
                class_vars[class_id] = var
            
            # Button frame
            btn_frame = ttk.Frame(config_window)
            btn_frame.pack(fill=tk.X, padx=15, pady=15)
            
            def on_ok():
                """Save the lane with selected vehicle types"""
                allowed_classes = [cid for cid, var in class_vars.items() if var.get()]
                
                if not allowed_classes:
                    messagebox.showerror("Error", "Please select at least one vehicle type!")
                    return
                
                try:
                    lane_id = len(self.traffic_lanes) + 1
                    lane_name = f"Lane {lane_id}"
                    direction = (1.0, 0.0)
                    
                    lane = TrafficLane(
                        lane_id=lane_id,
                        name=lane_name,
                        polygon=polygon_result[0],
                        allowed_classes=allowed_classes,
                        direction_vector=direction
                    )
                    
                    self.traffic_lanes[lane_id] = lane
                    self.violation_detector.add_lane(lane)
                    
                    # Redraw first frame
                    self._display_frame_on_canvas(frame)
                    
                    messagebox.showinfo("Success!", f"Lane {lane_id} created successfully!")
                    self.update_lane_info_display()
                    config_window.destroy()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to create lane: {str(e)}")
            
            ttk.Button(btn_frame, text="✓ OK - Save Lane", command=on_ok).pack(side=tk.LEFT, padx=5)
            ttk.Button(btn_frame, text="✗ Cancel", command=config_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Create ROI selector to draw 4 points
        roi_selector = ROISelector(frame, self.canvas, on_polygon_finished)
    
    def delete_traffic_lane(self):
        """Delete a traffic lane"""
        if len(self.traffic_lanes) == 0:
            messagebox.showerror("Error", "No lanes to delete!")
            return
        
        select_window = tk.Toplevel(self.root)
        select_window.title("Select Lane to Delete")
        select_window.geometry("300x250")
        select_window.resizable(False, False)
        
        ttk.Label(select_window, text="Select a lane to delete:", 
                 font=("Arial", 10, "bold")).pack(pady=10)
        
        listbox = tk.Listbox(select_window, height=8, font=("Arial", 9))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for lane_id, lane in self.traffic_lanes.items():
            listbox.insert(tk.END, f"Lane {lane_id}: {lane.name}")
        
        def on_delete():
            try:
                idx = listbox.curselection()[0]
                selected_lane_id = list(self.traffic_lanes.keys())[idx]
                
                if messagebox.askyesno("Confirm", f"Delete Lane {selected_lane_id}?"):
                    del self.traffic_lanes[selected_lane_id]
                    if selected_lane_id in self.violation_detector.lanes:
                        del self.violation_detector.lanes[selected_lane_id]
                    select_window.destroy()
                    messagebox.showinfo("Success", f"Lane {selected_lane_id} deleted!")
                    self.update_lane_info_display()
            except IndexError:
                messagebox.showerror("Error", "Please select a lane!")
        
        ttk.Button(select_window, text="Delete Selected", command=on_delete).pack(pady=10)
    
    def modify_traffic_lane(self):
        """Modify an existing lane"""
        if len(self.traffic_lanes) == 0:
            messagebox.showerror("Error", "No lanes to modify!")
            return
        
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video first!")
            return
        
        # Create dialog to select which lane to modify
        select_window = tk.Toplevel(self.root)
        select_window.title("Select Lane to Modify")
        select_window.geometry("300x250")
        select_window.resizable(False, False)
        
        ttk.Label(select_window, text="Select a lane to modify:", 
                 font=("Arial", 10, "bold")).pack(pady=10)
        
        # Listbox for lanes
        listbox = tk.Listbox(select_window, height=8, font=("Arial", 9))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        for lane_id, lane in self.traffic_lanes.items():
            listbox.insert(tk.END, f"Lane {lane_id}: {lane.name}")
        
        def on_select():
            try:
                idx = listbox.curselection()[0]
                selected_lane_id = list(self.traffic_lanes.keys())[idx]
                select_window.destroy()
                self._edit_lane(selected_lane_id)
            except IndexError:
                messagebox.showerror("Error", "Please select a lane!")
        
        ttk.Button(select_window, text="Modify Selected", command=on_select).pack(pady=10)
    
    def _edit_lane(self, lane_id):
        """Edit vehicle types for a specific lane"""
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video first!")
            return
        
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            messagebox.showerror("Error", "Cannot read video")
            return
        
        lane = self.traffic_lanes[lane_id]
        
        # Create dialog for vehicle type selection
        edit_window = tk.Toplevel(self.root)
        edit_window.title(f"Modify Lane {lane_id}")
        edit_window.geometry("400x400")
        edit_window.resizable(False, False)
        
        ttk.Label(edit_window, text=f"Lane {lane_id}: Edit Allowed Vehicles", 
                 font=("Arial", 10, "bold")).pack(pady=10)
        
        # Checkboxes for vehicle types
        class_vars = {}
        cb_frame = ttk.Frame(edit_window)
        cb_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        for class_id, class_name in sorted(self.detector.class_names.items()):
            var = tk.BooleanVar(value=class_id in lane.allowed_classes)
            cb = ttk.Checkbutton(cb_frame, text=f"{class_name} (ID:{class_id})", variable=var)
            cb.pack(anchor=tk.W, pady=3)
            class_vars[class_id] = var
        
        def on_save():
            allowed_classes = [cid for cid, var in class_vars.items() if var.get()]
            
            if not allowed_classes:
                messagebox.showerror("Error", "Please select at least one vehicle type!")
                return
            
            try:
                lane.allowed_classes = allowed_classes
                self.violation_detector.add_lane(lane)
                edit_window.destroy()
                messagebox.showinfo("Success", f"Lane {lane_id} updated successfully!")
                self.update_lane_info_display()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update lane: {str(e)}")
        
        btn_frame = ttk.Frame(edit_window)
        btn_frame.pack(fill=tk.X, padx=15, pady=15)
        ttk.Button(btn_frame, text="✓ Save Changes", command=on_save).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="✗ Cancel", command=edit_window.destroy).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def update_lane_info_display(self):
        """Update lane info display in settings"""
        if len(self.traffic_lanes) == 0:
            self.lane_info_label.config(text="No lanes configured", foreground="orange")
        else:
            lanes_text = f"{len(self.traffic_lanes)} lane(s):\n"
            for lane_id, lane in self.traffic_lanes.items():
                lanes_text += f"• Lane {lane_id}: {lane.name}\n"
            self.lane_info_label.config(text=lanes_text.strip(), foreground="green")
    
    def _display_frame_on_canvas(self, frame):
        """Display BGR frame on canvas"""
        try:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                h, w = frame_rgb.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                
                img = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image=img)
                
                self.canvas.delete("all")
                self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                       image=photo, anchor=tk.CENTER)
                self.canvas.image = photo
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def start_processing(self):
        """Start video processing in separate thread"""
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first!")
            return
        
        if not Path(self.video_path).exists():
            messagebox.showerror("Error", f"Video file not found: {self.video_path}")
            return
        
        if not Path(self.model_path).exists():
            messagebox.showerror("Error", 
                f"YOLO model not found!\nExpected: {self.model_path}")
            return
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.is_processing = True
        self.should_stop = False
        
        STrack.track_id_count = 0
        
        thread = threading.Thread(target=self.process_video, daemon=True)
        thread.start()
        
        self.update_display()
    
    def stop_processing(self):
        """Stop video processing"""
        self.should_stop = True
    
    def process_video(self):
        """Process video with ByteTrack (runs in separate thread)"""
        try:
            print("="*60)
            print("Initializing YOLO11s Traffic detector...")
            self.detector = YOLODetector(
                model_path=self.model_path,
                conf_threshold=self.det_conf_var.get(),
                device=self.device_var.get(),
                min_box_area=self.min_box_area_var.get(),
                edge_margin=self.edge_margin_var.get()
            )
            print(f"✓ YOLO11s Traffic model loaded")
            
            print("Initializing ByteTrack...")
            self.tracker = BYTETracker(
                track_buffer=self.track_buffer_var.get(),
                min_hits=3
            )
            print("✓ ByteTrack initialized")
            
            print(f"Opening video: {Path(self.video_path).name}")
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                print("❌ ERROR: Cannot open video file!")
                return
            
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")
            
            writer = None
            if self.save_output_var.get() and self.output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
                print(f"✓ Output will be saved to: {Path(self.output_path).name}")
            
            print("="*60)
            print("Processing started...")
            print(f"{'Frame':<10} {'Detections':<12} {'Tracks':<10} {'FPS':<10}")
            print("-"*50)
            
            frame_id = 0
            fps_list = []
            
            while not self.should_stop:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_id += 1
                start_time = time.time()
                
                detections = self.detector.detect(frame)
                img_shape = (height, width)
                online_tracks = self.tracker.update(detections, img_shape)
                
                if self.enable_violation_var.get() and len(self.traffic_lanes) > 0:
                    violations = self.violation_detector.detect_violations(online_tracks, frame_id)
                    for track in online_tracks:
                        if track.track_id in violations:
                            track.violation_type = violations[track.track_id]
                        else:
                            track.violation_type = ViolationType.NONE
                
                elapsed = time.time() - start_time
                fps_val = 1.0 / elapsed if elapsed > 0 else 0
                fps_list.append(fps_val)
                
                if frame_id % 30 == 0 or frame_id == 1:
                    print(f"{frame_id:<10} {len(detections):<12} "
                          f"{len(online_tracks):<10} {fps_val:<10.2f}")
                
                frame_vis = self.draw_tracks(frame, online_tracks, detections)
                
                info_text = [
                    f"Frame: {frame_id}/{total_frames}",
                    f"Detections: {len(detections)}",
                    f"Tracks: {len(online_tracks)}",
                    f"FPS: {fps_val:.1f}"
                ]
                
                y_offset = 30
                for text in info_text:
                    cv2.putText(frame_vis, text, (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30
                
                progress = (frame_id / total_frames) * 100
                self.progress_var.set(progress)
                self.status_label.config(
                    text=f"Processing: Frame {frame_id}/{total_frames} "
                         f"({progress:.1f}%) - FPS: {fps_val:.1f}")
                
                if not self.frame_queue.full():
                    try:
                        self.frame_queue.put_nowait(frame_vis)
                    except queue.Full:
                        pass
                
                if writer:
                    writer.write(frame_vis)
            
            self.cap.release()
            if writer:
                writer.release()
            
            print("\n" + "="*60)
            if self.should_stop:
                print("✓ Processing stopped by user")
            else:
                print("✓ Processing completed!")
            print(f"  Total frames processed: {frame_id}")
            if len(fps_list) > 0:
                print(f"  Average FPS: {np.mean(fps_list):.2f}")
            print(f"  Total tracks created: {STrack.track_id_count}")
            if self.output_path and writer:
                print(f"  Output saved to: {self.output_path}")
            print("="*60)
            
            self.status_label.config(text="✅ Processing completed!")
            self.progress_var.set(100)
            
            if not self.should_stop:
                messagebox.showinfo("Success", 
                                   f"Video processing completed!\n\n"
                                   f"Frames processed: {frame_id}\n"
                                   f"Average FPS: {np.mean(fps_list):.1f}\n"
                                   f"Total tracks: {STrack.track_id_count}")
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            
        finally:
            self.is_processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def draw_tracks(self, frame, tracks, detections):
        """Draw tracking results on frame"""
        for track in tracks:
            if not track.is_activated:
                continue
            
            tlbr = track.tlbr
            x1, y1, x2, y2 = map(int, tlbr)
            track_id = track.track_id
            
            if hasattr(track, 'violation_type') and track.violation_type != ViolationType.NONE:
                color = (0, 0, 255)
                line_thickness = 3
            else:
                color = (0, 255, 0)
                line_thickness = 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)
            
            class_name = "unknown"
            class_id_display = "?"
            if track.class_id is not None:
                class_name = self.detector.get_class_name(track.class_id)
                class_id_display = str(int(track.class_id))
            
            label = f"ID:{track_id} [{class_id_display}] {class_name}"
            if hasattr(track, 'violation_type') and track.violation_type != ViolationType.NONE:
                label += f" | VIOLATION"
            
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (255, 255, 255), 2)
        
        # Draw lane polygons
        if len(self.traffic_lanes) > 0:
            for lane_id, lane in self.traffic_lanes.items():
                polygon = np.array(lane.polygon, dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [polygon], (0, 255, 0))
                cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
                cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)
                cv2.putText(frame, f"Lane {lane_id}", 
                           (polygon[0][0], polygon[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def update_display(self):
        """Update canvas with latest frame"""
        if self.is_processing:
            try:
                frame = self.frame_queue.get_nowait()
                
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    h, w = frame_rgb.shape[:2]
                    scale = min(canvas_width / w, canvas_height / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                    
                    img = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(image=img)
                    
                    self.canvas.delete("all")
                    self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                           image=photo, anchor=tk.CENTER)
                    self.canvas.image = photo
                    
            except queue.Empty:
                pass
            
            self.root.after(30, self.update_display)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = ByteTrackGUI(root)
    
    print("=" * 60)
    print("ByteTrack GUI Started")
    print("=" * 60)
    
    root.mainloop()


if __name__ == '__main__':
    main()
