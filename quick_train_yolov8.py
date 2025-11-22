import os
import random
import shutil
from ultralytics import YOLO

# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # Install required packages if needed
    # Uncomment if packages are not installed
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    
    print("Setting up quick YOLOv8 training for human and car detection...")
    
    # Create dataset directories
    os.makedirs("dataset/images/train", exist_ok=True)
    os.makedirs("dataset/images/val", exist_ok=True)
    os.makedirs("dataset/labels/train", exist_ok=True)
    os.makedirs("dataset/labels/val", exist_ok=True)
    
    # Sample a small subset of images for quick training
    test_dir = os.path.join("test", "test")
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
    
    # Use only 50 images for faster training
    sample_size = min(50, len(image_files))
    sampled_images = random.sample(image_files, sample_size)
    
    # Split into train and validation sets (80/20)
    split_idx = int(len(sampled_images) * 0.8)
    train_images = sampled_images[:split_idx]
    val_images = sampled_images[split_idx:]
    
    print(f"Using {len(train_images)} images for training and {len(val_images)} for validation")
    
    # Copy images to train and validation directories
    for img in train_images:
        shutil.copy(os.path.join(test_dir, img), os.path.join("dataset/images/train", img))
        
        # Create synthetic labels for training (person=0, car=1)
        with open(os.path.join("dataset/labels/train", os.path.splitext(img)[0] + ".txt"), "w") as f:
            # Add a person annotation (50% chance)
            if random.random() > 0.5:
                x, y, w, h = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8), random.uniform(0.1, 0.3), random.uniform(0.2, 0.5)
                f.write(f"0 {x} {y} {w} {h}\n")
            
            # Add a car annotation (50% chance)
            if random.random() > 0.5:
                x, y, w, h = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8), random.uniform(0.1, 0.4), random.uniform(0.1, 0.3)
                f.write(f"1 {x} {y} {w} {h}\n")
    
    for img in val_images:
        shutil.copy(os.path.join(test_dir, img), os.path.join("dataset/images/val", img))
        
        # Create synthetic labels for validation
        with open(os.path.join("dataset/labels/val", os.path.splitext(img)[0] + ".txt"), "w") as f:
            if random.random() > 0.5:
                x, y, w, h = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8), random.uniform(0.1, 0.3), random.uniform(0.2, 0.5)
                f.write(f"0 {x} {y} {w} {h}\n")
            
            if random.random() > 0.5:
                x, y, w, h = random.uniform(0.2, 0.8), random.uniform(0.2, 0.8), random.uniform(0.1, 0.4), random.uniform(0.1, 0.3)
                f.write(f"1 {x} {y} {w} {h}\n")
    
    # Create YAML configuration
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
    
    print("Dataset and configuration prepared. Starting training...")
    
    # Load a pre-trained YOLOv8 nano model
    model = YOLO('yolov8n.pt')
    
    # Train the model with minimal settings for quick results
    model.train(
        data='dataset.yaml',
        epochs=1,  # Just 1 epoch for demonstration
        batch=4,
        imgsz=320,  # Smaller image size
        device='cpu',
        patience=1,
        save=True
    )
    
    print("Training complete. Testing the model...")
    
    # Test the model on a few images
    for i, img in enumerate(val_images[:3]):  # Test on 3 validation images
        results = model(os.path.join("dataset/images/val", img))
        
        # Save the detection results
        os.makedirs("results", exist_ok=True)
        save_path = os.path.join("results", f"detection_{i}.jpg")
        results[0].save(save_path)
        print(f"Detection result saved to {save_path}")
    
    print("YOLOv8 model training and testing completed!")

if __name__ == "__main__":
    main()