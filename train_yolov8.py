import os
import shutil
import cv2
import numpy as np
from PIL import Image
import subprocess
import sys

# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Install required packages
def setup_environment():
    print("Setting up environment...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "opencv-python", "pillow"])
    print("Environment setup complete.")

# Create YOLO dataset structure
def prepare_dataset_structure():
    print("Preparing dataset structure...")
    
    # Create directories if they don't exist
    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/images/val", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)
    os.makedirs("dataset/labels/val", exist_ok=True)
    
    print("Dataset structure created.")

# Resize images to appropriate dimensions for YOLOv8
def resize_images(source_dir, target_dir, target_size=(640, 640)):
    print(f"Resizing images from {source_dir} to {target_dir}...")
    
    # Get list of image files
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        source_path = os.path.join(source_dir, img_file)
        target_path = os.path.join(target_dir, img_file)
        
        # Open and resize image
        img = Image.open(source_path)
        img_resized = img.resize(target_size, Image.LANCZOS)
        img_resized.save(target_path)
    
    print(f"Resized {len(image_files)} images.")

# Create dummy annotations for human and car detection
def create_annotations(image_dir, label_dir):
    print(f"Creating annotations for images in {image_dir}...")
    
    # Class mapping: 0 = person, 1 = car
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        
        # Open image to get dimensions
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        
        # For demonstration, we'll create synthetic annotations
        # In a real scenario, you would use actual annotations
        
        # Randomly decide if image has person, car, both or none
        has_person = np.random.random() > 0.3
        has_car = np.random.random() > 0.3
        
        with open(label_path, 'w') as f:
            if has_person:
                # Format: class x_center y_center width height (normalized 0-1)
                x_center = np.random.uniform(0.2, 0.8)
                y_center = np.random.uniform(0.2, 0.8)
                bbox_width = np.random.uniform(0.1, 0.3)
                bbox_height = np.random.uniform(0.2, 0.5)
                f.write(f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n")
            
            if has_car:
                x_center = np.random.uniform(0.2, 0.8)
                y_center = np.random.uniform(0.2, 0.8)
                bbox_width = np.random.uniform(0.1, 0.4)
                bbox_height = np.random.uniform(0.1, 0.3)
                f.write(f"1 {x_center} {y_center} {bbox_width} {bbox_height}\n")
    
    print(f"Created annotations for {len(image_files)} images.")

# Split data into train and validation sets
def split_data(test_dir, train_ratio=0.8):
    print("Splitting data into train and validation sets...")
    
    # Get list of image files
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    np.random.shuffle(image_files)
    
    # Split into train and validation
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Copy images to train and validation directories
    for img_file in train_files:
        shutil.copy(
            os.path.join(test_dir, img_file),
            os.path.join("dataset/images/train", img_file)
        )
    
    for img_file in val_files:
        shutil.copy(
            os.path.join(test_dir, img_file),
            os.path.join("dataset/images/val", img_file)
        )
    
    print(f"Split {len(train_files)} images for training and {len(val_files)} for validation.")
    return train_files, val_files

# Create YAML configuration file for YOLOv8
def create_yaml_config():
    print("Creating YAML configuration file...")
    
    yaml_content = """
# Dataset configuration for YOLOv8
path: ./dataset  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
  0: person
  1: car
"""
    
    with open("dataset.yaml", "w") as f:
        f.write(yaml_content)
    
    print("YAML configuration file created.")

# Train YOLOv8 model
def train_model(epochs=3, batch_size=4, img_size=320):
    print("Training YOLOv8 model...")
    
    from ultralytics import YOLO
    
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    
    # Train the model with CPU - using smaller parameters for faster training
    model.train(
        data='dataset.yaml',
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device='cpu',
        patience=2,
        save=True
    )
    
    print("Model training complete.")
    return model

# Evaluate the model
def evaluate_model(model):
    print("Evaluating model...")
    
    # Evaluate the model
    metrics = model.val()
    
    print("Model evaluation complete.")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    
    return metrics

def main():
    # Set up environment
    setup_environment()
    
    # Prepare dataset structure
    prepare_dataset_structure()
    
    # Split data into train and validation sets
    test_dir = os.path.join("test", "test")
    train_files, val_files = split_data(test_dir)
    
    # Resize images
    resize_images("dataset/images/train", "dataset/images/train")
    resize_images("dataset/images/val", "dataset/images/val")
    
    # Create annotations
    create_annotations("dataset/images/train", "dataset/labels/train")
    create_annotations("dataset/images/val", "dataset/labels/val")
    
    # Create YAML configuration file
    create_yaml_config()
    
    # Train model
    model = train_model(epochs=5)  # Reduced epochs for faster training
    
    # Evaluate model
    evaluate_model(model)
    
    print("YOLOv8 training pipeline completed successfully!")

if __name__ == "__main__":
    main()