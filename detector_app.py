import os
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from ultralytics import YOLO

# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class DetectorApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Human and Car Detector")
        self.root.geometry("1000x700")
        
        # Load the YOLOv8 model
        self.model = YOLO(model_path)
        
        # Initialize camera variables
        self.camera = None
        self.is_camera_on = False
        self.camera_thread = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Upload image button
        upload_btn = ttk.Button(control_frame, text="Upload Image", command=self.upload_image)
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Camera control button
        self.camera_btn = ttk.Button(control_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Detect button
        detect_btn = ttk.Button(control_frame, text="Detect", command=self.detect_objects)
        detect_btn.pack(side=tk.LEFT, padx=5)
        
        # Image display area
        self.canvas = tk.Canvas(main_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Detection results area
        results_frame = ttk.LabelFrame(main_frame, text="Detection Results")
        results_frame.pack(fill=tk.X, pady=10)
        
        self.result_text = tk.Text(results_frame, height=5, wrap=tk.WORD)
        self.result_text.pack(fill=tk.X, padx=5, pady=5)
        
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            # Stop camera if it's running
            if self.is_camera_on:
                self.toggle_camera()
            
            self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
            
            # Load and display the image
            self.current_image = cv2.imread(file_path)
            self.display_image(self.current_image)
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if self.is_camera_on:
            # Turn off camera
            self.is_camera_on = False
            self.camera_btn.config(text="Start Camera")
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            self.status_var.set("Camera stopped")
        else:
            # Turn on camera
            self.is_camera_on = True
            self.camera_btn.config(text="Stop Camera")
            self.status_var.set("Camera started")
            
            # Start camera in a separate thread
            self.camera_thread = threading.Thread(target=self.camera_stream)
            self.camera_thread.daemon = True
            self.camera_thread.start()
    
    def camera_stream(self):
        """Handle camera stream"""
        self.camera = cv2.VideoCapture(1)
        
        if not self.camera.isOpened():
            self.status_var.set("Error: Could not open any camera")
            self.is_camera_on = False
            self.camera_btn.config(text="Start Camera")
            return
        
        while self.is_camera_on:
            ret, frame = self.camera.read()
            if ret:
                self.current_image = frame
                self.display_image(frame)
                # Add a small delay to reduce CPU usage
                self.root.after(10)
            else:
                break
                
        # Clean up
        if self.camera is not None:
            self.camera.release()
    
    def display_image(self, cv_image):
        """Display an OpenCV image on the canvas"""
        if cv_image is None:
            return
            
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Use fixed dimensions for display to prevent jumping
        fixed_width = 800
        fixed_height = 600
        
        # Get image dimensions
        img_height, img_width = image_rgb.shape[:2]
        
        # Calculate scaling factor to fit within fixed dimensions
        scale = min(fixed_width/img_width, fixed_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        resized_image = cv2.resize(image_rgb, (new_width, new_height))
        
        # Create a black background of fixed size
        display_image = np.zeros((fixed_height, fixed_width, 3), dtype=np.uint8)
        
        # Calculate position to center the image
        y_offset = (fixed_height - new_height) // 2
        x_offset = (fixed_width - new_width) // 2
        
        # Place the resized image on the black background
        display_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(display_image))
        
        # Update canvas with fixed dimensions
        if not hasattr(self, 'canvas_initialized'):
            self.canvas.config(width=fixed_width, height=fixed_height)
            self.canvas_initialized = True
            
        # Clear previous content and display new image
        self.canvas.delete("all")
        self.canvas.create_image(fixed_width//2, fixed_height//2, image=self.photo, anchor=tk.CENTER)
        
    def detect_objects(self):
        """Detect humans and cars in the current image"""
        if not hasattr(self, 'current_image') or self.current_image is None:
            self.status_var.set("No image to detect objects")
            return
            
        self.status_var.set("Detecting objects...")
        
        # Make a copy of the current image for drawing
        image_copy = self.current_image.copy()
        
        # Run detection
        results = self.model(image_copy)
        
        # Clear previous results
        self.result_text.delete(1.0, tk.END)
        
        # Process results
        detection_count = {"person": 0, "car": 0}
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Map class index to name (using the model's class names)
                class_names = self.model.names
                class_name = class_names[cls] if cls in class_names else "unknown"
                
                # For display purposes, map to our target classes
                display_name = "Human" if class_name == "person" else "Car" if class_name == "car" else class_name
                
                # Update count
                if class_name in detection_count:
                    detection_count[class_name] += 1
                
                # Draw bounding box
                color = (0, 255, 0) if class_name == "person" else (0, 0, 255)  # Green for person, Red for car
                cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{display_name}: {conf:.2f}"
                cv2.putText(image_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display the image with detections
        self.display_image(image_copy)
        
        # Update results text
        result_str = f"Detection Results:\n"
        result_str += f"- Persons detected: {detection_count['person']}\n"
        result_str += f"- Cars detected: {detection_count['car']}\n"
        self.result_text.insert(tk.END, result_str)
        
        self.status_var.set("Detection completed")

def main():
    # Set the model path
    model_path = "d:\\l\\archive\\yolov8n.pt"
    
    # Create the main window
    root = tk.Tk()
    app = DetectorApp(root, model_path)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()