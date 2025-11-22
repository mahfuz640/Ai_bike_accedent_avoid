import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import random

# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def test_model(model_path, test_dir, output_dir="detection_results", num_images=5):
    """
    Test the trained YOLOv8 model on sample images
    
    Args:
        model_path: Path to the trained model weights
        test_dir: Directory containing test images
        output_dir: Directory to save detection results
        num_images: Number of test images to process
    """
    print(f"Testing model {model_path} on images from {test_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    model = YOLO(model_path)
    
    # Get list of image files
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Select random images if there are more than num_images
    if len(image_files) > num_images:
        image_files = random.sample(image_files, num_images)
    
    # Class names
    class_names = {0: 'person', 1: 'car'}
    
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        
        # Run inference
        results = model(img_path)
        
        # Get the original image
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except:
            font = ImageFont.load_default()
        
        # Process detection results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                
                # Get confidence and class
                conf = box.conf[0]
                cls = int(box.cls[0])
                
                # Draw bounding box
                color = (255, 0, 0) if cls == 0 else (0, 0, 255)  # Red for person, Blue for car
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label
                label = f"{class_names[cls]}: {conf:.2f}"
                text_size = draw.textbbox((0, 0), label, font=font)
                text_width = text_size[2] - text_size[0]
                text_height = text_size[3] - text_size[1]
                
                draw.rectangle([x1, y1 - text_height - 4, x1 + text_width, y1], fill=color)
                draw.text((x1, y1 - text_height - 4), label, fill=(255, 255, 255), font=font)
        
        # Save the image with detections
        output_path = os.path.join(output_dir, f"detection_{img_file}")
        img.save(output_path)
        print(f"Saved detection result to {output_path}")

def main():
    # Path to the trained model
    model_path = "runs/detect/train/weights/best.pt"
    
    # Path to test images
    test_dir = os.path.join("test", "test")
    
    # Test the model
    test_model(model_path, test_dir)
    
    print("Testing completed!")

if __name__ == "__main__":
    main()