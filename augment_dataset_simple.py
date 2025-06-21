import os
import random
from PIL import Image, ImageEnhance
from pathlib import Path
import shutil

def create_augmented_dataset():
    """
    Create augmented dataset using Keras ImageDataGenerator with:
    - Horizontal flipping
    - Vertical flipping  
    - Brightness change range [0.2:1.2]
    
    Creates exactly 3 augmented versions of each image plus the original.
    """
    
    # Define paths
    input_dir = Path('filtered_dataset')
    output_dir = Path('augmented_dataset')
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    # Get list of all images in filtered dataset
    image_files = list(input_dir.glob('*.jpg'))
    total_images = len(image_files)
    
    if total_images == 0:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {total_images} images in filtered dataset")
    print("Starting augmentation process...")
    print("Creating 3 augmented versions + 1 original copy per image")
    
    # Statistics
    processed_count = 0
    augmented_count = 0
    
    # Process each image
    for i, img_path in enumerate(image_files):
        try:
            # Load image
            img = Image.open(img_path)
            
            # Extract base filename (without extension)
            base_filename = img_path.stem
            
            # Copy original image first
            original_output_path = output_dir / f"{base_filename}_original.jpg"
            img.save(original_output_path, 'JPEG', quality=95)
            
            # Create 3 augmented versions
            augmentations = [
                ("hflip", img.transpose(Image.FLIP_LEFT_RIGHT)),  # Horizontal flip
                ("vflip", img.transpose(Image.FLIP_TOP_BOTTOM)),  # Vertical flip
                ("bright", ImageEnhance.Brightness(img).enhance(random.uniform(0.2, 1.2)))  # Brightness
            ]
            
            for aug_name, aug_img in augmentations:
                output_filename = f"{base_filename}_aug_{aug_name}.jpg"
                output_path = output_dir / output_filename
                aug_img.save(output_path, 'JPEG', quality=95)
                augmented_count += 1
            
            processed_count += 1
            
            # Progress update every 1000 images
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count}/{total_images} images...")
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    print(f"\n✅ Augmentation completed successfully!")
    print(f"📁 Original images processed: {processed_count}")
    print(f"🔄 Augmented images created: {augmented_count}")
    print(f"📊 Total images in augmented dataset: {processed_count + augmented_count}")
    print(f"💾 Images saved in: {output_dir}")
    
    # Calculate and display storage information
    try:
        total_size = sum(f.stat().st_size for f in output_dir.glob('*.jpg')) / (1024 * 1024)
        print(f"💽 Total size of augmented dataset: {total_size:.2f} MB")
        
        # Expected total (4x original size since we have original + 3 augmented)
        original_size = sum(f.stat().st_size for f in input_dir.glob('*.jpg')) / (1024 * 1024)
        print(f"📈 Size increase factor: {total_size/original_size:.1f}x")
        
    except Exception as e:
        print(f"Could not calculate directory size: {e}")
    
    # Display class distribution info
    print(f"\n📋 Augmentation Summary:")
    print(f"   • Each original image → 4 total images (1 original + 3 augmented)")
    print(f"   • Augmentations applied: horizontal flip, vertical flip, brightness [0.2-1.2]")
    print(f"   • Expected total images: {total_images * 4}")
    print(f"   • Actual total images: {processed_count + augmented_count}")

if __name__ == "__main__":
    create_augmented_dataset() 