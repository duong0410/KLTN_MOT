"""
GUI Application for ByteTrack-YOLO
Tkinter-based interface for video tracking
"""

import sys
import time
import queue
import threading
import traceback
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext, messagebox
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as e:
    print(f"Error: GUI requires additional packages: {e}")
    print("Install with: pip install pillow opencv-python")
    sys.exit(1)

from ..detector import YOLODetector
from ..tracker import BYTETracker, STrack
from ..utils.visualization import TrackVisualizer


class ByteTrackGUI:
    """GUI Application for ByteTrack + YOLO video tracking"""
    
    def __init__(self, root, default_model_path=None):
        """
        Initialize GUI
        
        Args:
            root: Tkinter root window
            default_model_path: Default path to YOLO model
        """
        self.root = root
        self.root.title("ByteTrack-YOLO Video Tracking")
        self.root.geometry("1400x900")
        
        # State variables
        self.video_path = None
        self.output_path = None
        self.is_processing = False
        self.should_stop = False
        self.cap = None
        self.tracker = None
        self.detector = None
        self.visualizer = None
        
        # Default model path
        if default_model_path:
            self.model_path = default_model_path
        else:
            self.model_path = "models/yolo11s_traffic.pt"
        
        # Queue for thread communication
        self.log_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Create GUI
        self.create_widgets()
        
        # Start log update loop
        self.update_logs()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Left Panel: Configuration with Scrollbar
        left_container = ttk.Frame(self.root)
        left_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(left_container, width=350, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding="5")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create window and bind width to canvas
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Make scrollable_frame expand to canvas width
        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", on_canvas_configure)
        
        # Grid canvas and scrollbar - scrollbar ngay sát canvas
        canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure left_container grid
        left_container.columnconfigure(0, weight=0)  # Canvas không expand
        left_container.rowconfigure(0, weight=1)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Use scrollable_frame as left_frame
        left_frame = scrollable_frame
        
        # Title
        title_label = ttk.Label(left_frame, text="ByteTrack Configuration", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Video Selection
        ttk.Label(left_frame, text="Video File:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky=tk.W, pady=5)
        
        video_frame = ttk.Frame(left_frame)
        video_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.video_label = ttk.Label(video_frame, text="No video selected", 
                                     foreground="gray", wraplength=300)
        self.video_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(video_frame, text="Browse...", command=self.browse_video).pack(
            side=tk.RIGHT, padx=(5, 0))
        
        # Model Path
        ttk.Label(left_frame, text="YOLO Model:", font=("Arial", 10, "bold")).grid(
            row=3, column=0, sticky=tk.W, pady=5)
        
        model_frame = ttk.Frame(left_frame)
        model_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        model_label = ttk.Label(model_frame, text=Path(self.model_path).name, 
                               foreground="blue", font=("Arial", 9))
        model_label.pack(side=tk.LEFT)
        
        # Device Selection
        ttk.Label(left_frame, text="Device:", font=("Arial", 10, "bold")).grid(
            row=5, column=0, sticky=tk.W, pady=5)
        
        self.device_var = tk.StringVar(value="cuda")
        device_frame = ttk.Frame(left_frame)
        device_frame.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        ttk.Radiobutton(device_frame, text="GPU (CUDA)", variable=self.device_var, 
                       value="cuda").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(device_frame, text="CPU", variable=self.device_var, 
                       value="cpu").pack(side=tk.LEFT)
        
        # Separator
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
            row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Detection Parameters
        ttk.Label(left_frame, text="Detection Parameters", 
                 font=("Arial", 11, "bold")).grid(row=8, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(5, 10))
        
        # YOLO Confidence
        ttk.Label(left_frame, text="YOLO Confidence:").grid(
            row=9, column=0, sticky=tk.W, pady=2)
        self.det_conf_var = tk.DoubleVar(value=0.01)
        det_conf_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, 
                                   variable=self.det_conf_var, orient=tk.HORIZONTAL)
        det_conf_scale.grid(row=9, column=1, sticky=(tk.W, tk.E), pady=2)
        self.det_conf_label = ttk.Label(left_frame, text="0.01")
        self.det_conf_label.grid(row=10, column=1, sticky=tk.W)
        det_conf_scale.config(command=lambda v: self.det_conf_label.config(
            text=f"{float(v):.2f}"))
        
        # Separator
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
            row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # ByteTrack Parameters
        ttk.Label(left_frame, text="ByteTrack Parameters", 
                 font=("Arial", 11, "bold")).grid(row=12, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(5, 10))
        
        # High Confidence Threshold
        ttk.Label(left_frame, text="High Confidence:").grid(
            row=13, column=0, sticky=tk.W, pady=2)
        self.det_conf_high_var = tk.DoubleVar(value=0.5)
        high_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, 
                              variable=self.det_conf_high_var, orient=tk.HORIZONTAL)
        high_scale.grid(row=13, column=1, sticky=(tk.W, tk.E), pady=2)
        self.high_label = ttk.Label(left_frame, text="0.50")
        self.high_label.grid(row=14, column=1, sticky=tk.W)
        high_scale.config(command=lambda v: self.high_label.config(text=f"{float(v):.2f}"))
        
        # Low Confidence Threshold
        ttk.Label(left_frame, text="Low Confidence:").grid(
            row=15, column=0, sticky=tk.W, pady=2)
        self.det_conf_low_var = tk.DoubleVar(value=0.1)
        low_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, 
                             variable=self.det_conf_low_var, orient=tk.HORIZONTAL)
        low_scale.grid(row=15, column=1, sticky=(tk.W, tk.E), pady=2)
        self.low_label = ttk.Label(left_frame, text="0.10")
        self.low_label.grid(row=16, column=1, sticky=tk.W)
        low_scale.config(command=lambda v: self.low_label.config(text=f"{float(v):.2f}"))
        
        # New Track Threshold
        ttk.Label(left_frame, text="New Track Threshold:").grid(
            row=17, column=0, sticky=tk.W, pady=2)
        self.new_track_var = tk.DoubleVar(value=0.6)
        new_track_scale = ttk.Scale(left_frame, from_=0.0, to=1.0, 
                                    variable=self.new_track_var, orient=tk.HORIZONTAL)
        new_track_scale.grid(row=17, column=1, sticky=(tk.W, tk.E), pady=2)
        self.new_track_label = ttk.Label(left_frame, text="0.60")
        self.new_track_label.grid(row=18, column=1, sticky=tk.W)
        new_track_scale.config(command=lambda v: self.new_track_label.config(
            text=f"{float(v):.2f}"))
        
        # Track Buffer
        ttk.Label(left_frame, text="Track Buffer (frames):").grid(
            row=19, column=0, sticky=tk.W, pady=2)
        self.track_buffer_var = tk.IntVar(value=30)
        buffer_spinbox = ttk.Spinbox(left_frame, from_=1, to=100, 
                                     textvariable=self.track_buffer_var, width=10)
        buffer_spinbox.grid(row=19, column=1, sticky=tk.W, pady=2)
        
        # Separator
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
            row=20, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Detection Filters
        ttk.Label(left_frame, text="Detection Filters", 
                 font=("Arial", 11, "bold")).grid(row=21, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(5, 10))
        
        # Min Box Area
        ttk.Label(left_frame, text="Min Box Area (pixels):").grid(
            row=22, column=0, sticky=tk.W, pady=2)
        self.min_box_area_var = tk.IntVar(value=400)
        area_spinbox = ttk.Spinbox(left_frame, from_=100, to=5000, increment=100,
                                   textvariable=self.min_box_area_var, width=10)
        area_spinbox.grid(row=22, column=1, sticky=tk.W, pady=2)
        ttk.Label(left_frame, text="(Filter tiny/distant boxes)", 
                 font=("Arial", 8), foreground="gray").grid(
            row=23, column=1, sticky=tk.W)
        
        # Edge Margin
        ttk.Label(left_frame, text="Edge Margin (pixels):").grid(
            row=24, column=0, sticky=tk.W, pady=2)
        self.edge_margin_var = tk.IntVar(value=10)
        margin_spinbox = ttk.Spinbox(left_frame, from_=0, to=100, increment=5,
                                     textvariable=self.edge_margin_var, width=10)
        margin_spinbox.grid(row=24, column=1, sticky=tk.W, pady=2)
        ttk.Label(left_frame, text="(Filter objects leaving frame)", 
                 font=("Arial", 8), foreground="gray").grid(
            row=25, column=1, sticky=tk.W)
        
        # Separator
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(
            row=26, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Output Settings
        ttk.Label(left_frame, text="Output Settings", 
                 font=("Arial", 11, "bold")).grid(row=27, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(5, 10))
        
        self.save_output_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_frame, text="Save output video", 
                       variable=self.save_output_var,
                       command=self.toggle_output).grid(row=28, column=0, 
                                                        columnspan=2, sticky=tk.W, pady=5)
        
        # Control Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=29, column=0, columnspan=2, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="▶ Start Processing", 
                                       command=self.start_processing,
                                       width=20)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="⏹ Stop", 
                                      command=self.stop_processing, 
                                      state=tk.DISABLED, width=15)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Configure grid weights for left frame
        left_frame.columnconfigure(1, weight=1)
        
        # Configure grid weights for main window
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        left_container.rowconfigure(0, weight=1)
        
        # Right Panel: Video Display and Logs
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Video Display
        video_label = ttk.Label(right_frame, text="📹 Video Preview", 
                               font=("Arial", 12, "bold"))
        video_label.pack(pady=(0, 5))
        
        self.canvas = tk.Canvas(right_frame, width=800, height=450, bg="black")
        self.canvas.pack(pady=(0, 10))
        
        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(right_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        # Status Label
        self.status_label = ttk.Label(right_frame, text="Ready to process", 
                                      font=("Arial", 10))
        self.status_label.pack(pady=(0, 10))
        
        # Logs
        log_label = ttk.Label(right_frame, text="📋 Processing Logs", 
                             font=("Arial", 11, "bold"))
        log_label.pack(pady=(0, 5))
        
        self.log_text = scrolledtext.ScrolledText(right_frame, height=15, 
                                                  width=90, state=tk.DISABLED,
                                                  font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
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
            self.log(f"✓ Video selected: {filename}")
            
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
            
    def log(self, message):
        """Add message to log"""
        self.log_queue.put(message)
        
    def update_logs(self):
        """Update log text widget from queue"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self.update_logs)
        
    def start_processing(self):
        """Start video processing in separate thread"""
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first!")
            return
        
        if not Path(self.video_path).exists():
            messagebox.showerror("Error", f"Video file not found: {self.video_path}")
            return
        
        # Check if model exists
        if not Path(self.model_path).exists():
            messagebox.showerror("Error", 
                f"YOLO model not found!\n\nExpected path:\n{self.model_path}")
            return
        
        # Disable controls
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.is_processing = True
        self.should_stop = False
        
        # Reset track counter
        STrack.track_id_count = 0
        
        # Start processing thread
        thread = threading.Thread(target=self.process_video, daemon=True)
        thread.start()
        
        # Start frame display update
        self.update_display()
        
    def stop_processing(self):
        """Stop video processing"""
        self.should_stop = True
        self.log("⚠ Stopping processing...")
        
    def process_video(self):
        """Process video with ByteTrack (runs in separate thread)"""
        try:
            # Initialize detector
            self.log("="*60)
            self.log("Initializing YOLO detector...")
            self.detector = YOLODetector(
                model_path=self.model_path,
                conf_threshold=self.det_conf_var.get(),
                device=self.device_var.get(),
                min_box_area=self.min_box_area_var.get(),
                edge_margin=self.edge_margin_var.get()
            )
            
            # Initialize tracker
            self.log("Initializing ByteTrack...")
            self.tracker = BYTETracker(
                det_conf_high=self.det_conf_high_var.get(),
                det_conf_low=self.det_conf_low_var.get(),
                new_track_thresh=self.new_track_var.get(),
                track_buffer=self.track_buffer_var.get()
            )
            self.log("✓ ByteTrack initialized")
            
            # Initialize visualizer
            self.visualizer = TrackVisualizer(
                class_names=self.detector.class_names
            )
            
            # Open video
            self.log(f"Opening video: {Path(self.video_path).name}")
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                self.log(" ERROR: Cannot open video file!")
                return
            
            # Get video properties
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.log(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
            
            # Setup video writer
            writer = None
            if self.save_output_var.get() and self.output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
                self.log(f" Output: {Path(self.output_path).name}")
            
            self.log("="*60)
            self.log("Processing started...")
            self.log(f"{'Frame':<10} {'Detections':<12} {'Tracks':<10} {'FPS':<10}")
            self.log("-"*50)
            
            # Process video
            frame_id = 0
            fps_list = []
            
            while not self.should_stop:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_id += 1
                start_time = time.time()
                
                # Run detection
                detections = self.detector.detect(frame)
                
                # Run tracking
                online_tracks = self.tracker.update(detections, (height, width))
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps_val = 1.0 / elapsed if elapsed > 0 else 0
                fps_list.append(fps_val)
                
                # Log progress
                if frame_id % 30 == 0 or frame_id == 1:
                    self.log(f"{frame_id:<10} {len(detections):<12} {len(online_tracks):<10} {fps_val:<10.2f}")
                
                # Draw tracks
                frame_vis = self.visualizer.draw_tracks(frame.copy(), online_tracks)
                frame_vis = self.visualizer.draw_info_panel(
                    frame_vis, frame_id, total_frames, 
                    len(detections), len(online_tracks), fps_val
                )
                
                # Update progress
                progress = (frame_id / total_frames) * 100
                self.progress_var.set(progress)
                self.status_label.config(text=f"Processing frame {frame_id}/{total_frames}")
                
                # Queue frame for display
                try:
                    self.frame_queue.put_nowait(frame_vis)
                except queue.Full:
                    pass
                
                # Save frame
                if writer:
                    writer.write(frame_vis)
            
            # Cleanup
            self.cap.release()
            if writer:
                writer.release()
            
            # Summary
            self.log("\n" + "="*60)
            if self.should_stop:
                self.log("Processing stopped by user")
            else:
                self.log("Processing completed!")
            self.log(f"  Total frames: {frame_id}")
            if len(fps_list) > 0:
                self.log(f"  Average FPS: {np.mean(fps_list):.2f}")
            self.log(f"  Total tracks: {STrack.track_id_count}")
            if self.output_path and writer:
                self.log(f"  Output saved: {self.output_path}")
            self.log("="*60)
            
            self.status_label.config(text="Processing completed!")
            self.progress_var.set(100)
            
            if not self.should_stop:
                messagebox.showinfo("Complete", "Video processing completed successfully!")
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            self.log("Traceback:")
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            
        finally:
            self.is_processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
    def update_display(self):
        """Update canvas with latest frame"""
        if self.is_processing:
            try:
                frame = self.frame_queue.get_nowait()
                
                # Resize frame to fit canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Resize maintaining aspect ratio
                    h, w = frame.shape[:2]
                    scale = min(canvas_width/w, canvas_height/h)
                    new_w, new_h = int(w*scale), int(h*scale)
                    frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
                    
                    # Convert to PhotoImage
                    img = Image.fromarray(frame_resized)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    # Update canvas
                    self.canvas.delete("all")
                    self.canvas.create_image(
                        canvas_width//2, canvas_height//2, 
                        image=imgtk, anchor=tk.CENTER
                    )
                    self.canvas.imgtk = imgtk
                    
            except queue.Empty:
                pass
            
            # Schedule next update
            self.root.after(30, self.update_display)


def create_gui(default_model_path=None):
    """
    Create and run GUI application
    
    Args:
        default_model_path: Default path to YOLO model
    """
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except:
        pass
    
    app = ByteTrackGUI(root, default_model_path)
    
    print("=" * 60)
    print("ByteTrack-YOLO GUI Started")
    print("=" * 60)
    
    root.mainloop()
