import os
import shutil
import random
from pathlib import Path

# Configuration
SOURCE_ROOT = Path("./dataset")
DEST_ROOT = Path("./dataset_final")
CLASSES = ["Histopathological", "Clinical"]
SPLIT_RATIO = 0.8
SEED = 42

def prepare_dataset():
    """
    Splits the manual dataset into a standard PyTorch ImageFolder structure (Train/Val).
    """
    random.seed(SEED)
    
    if not SOURCE_ROOT.exists():
        print(f"Error: Source directory '{SOURCE_ROOT}' not found. Please create it and add your '{CLASSES[0]}' and '{CLASSES[1]}' folders.")
        return

    # Clean destination if it exists to ensure a fresh split
    if DEST_ROOT.exists():
        print(f"Cleaning existing destination '{DEST_ROOT}'...")
        shutil.rmtree(DEST_ROOT)
    
    # Create directory structure
    for split in ['train', 'val']:
        for class_name in CLASSES:
            (DEST_ROOT / split / class_name).mkdir(parents=True, exist_ok=True)

    print("Processing images...")
    
    total_images = 0
    
    for class_name in CLASSES:
        class_dir = SOURCE_ROOT / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory '{class_dir}' not found. Skipping.")
            continue
            
        # Gather all valid image files
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        random.shuffle(images)
        
        split_idx = int(len(images) * SPLIT_RATIO)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        print(f"Class '{class_name}': Found {len(images)} images. Split -> {len(train_imgs)} Train, {len(val_imgs)} Val.")
        
        # Copy files
        for img in train_imgs:
            shutil.copy2(class_dir / img, DEST_ROOT / 'train' / class_name / img)
            
        for img in val_imgs:
            shutil.copy2(class_dir / img, DEST_ROOT / 'val' / class_name / img)
            
        total_images += len(images)

    if total_images == 0:
        print("No images found. Please check your './dataset_source' folder structure.")
    else:
        print(f"Dataset preparation complete. Data organized in '{DEST_ROOT}'.")

if __name__ == "__main__":
    prepare_dataset()
