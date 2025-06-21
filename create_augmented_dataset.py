import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from pathlib import Path
import json
from tqdm import tqdm

def create_augmented_dataset():
    """
    Create augmented dataset using Keras ImageDataGenerator with:
    - Horizontal flipping
    - Vertical flipping  
    - Brightness change range [0.2:1.2]
    """
    
    # Define paths
    input_dir = Path('filtered_dataset')
    output_dir = Path('augmented_dataset')
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Configure ImageDataGenerator with specified augmentations
    datagen = ImageDataGenerator(
        horizontal_flip=True,        # Horizontal flipping
        vertical_flip=True,          # Vertical flipping
        brightness_range=[0.2, 1.2], # Brightness change range [0.2:1.2]
        fill_mode='nearest'          # Fill mode for any empty pixels
    )
    
    # Get list of all images in filtered dataset
    image_files = list(input_dir.glob('*.jpg'))
    total_images = len(image_files)
    
    print(f"Found {total_images} images in filtered dataset")
    print("Starting augmentation process...")
    
    # Statistics
    processed_count = 0
    augmented_count = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc="Augmenting images"):
        try:
            # Load image
            img = load_img(img_path, target_size=None)  # Keep original size
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Extract base filename (without extension)
            base_filename = img_path.stem  # e.g., "UnsharpMask.123.Cell"
            
            # Generate augmented images
            # We'll create multiple augmented versions of each image
            augmentation_count = 0
            
            for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
                # Convert back to PIL Image
                augmented_img = array_to_img(batch[0])
                
                # Create output filename
                output_filename = f"{base_filename}_aug{i+1}.jpg"
                output_path = output_dir / output_filename
                
                # Save augmented image
                augmented_img.save(output_path, 'JPEG', quality=95)
                
                augmentation_count += 1
                augmented_count += 1
                
                # Generate 5 augmented versions per original image
                if augmentation_count >= 5:
                    break
            
            # Also copy the original image to augmented dataset
            original_output_path = output_dir / f"{base_filename}_original.jpg"
            img.save(original_output_path, 'JPEG', quality=95)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    print(f"\nAugmentation completed!")
    print(f"Original images processed: {processed_count}")
    print(f"Augmented images created: {augmented_count}")
    print(f"Total images in augmented dataset: {processed_count + augmented_count}")
    print(f"Images saved in: {output_dir}")
    
    # Calculate and display storage information
    try:
        total_size = sum(f.stat().st_size for f in output_dir.glob('*.jpg')) / (1024 * 1024)
        print(f"Total size of augmented dataset: {total_size:.2f} MB")
    except Exception as e:
        print(f"Could not calculate directory size: {e}")

def create_augmented_dataset_with_class_balance():
    """
    Alternative version that maintains class balance during augmentation
    """
    
    # Define paths
    input_dir = Path('filtered_dataset')
    output_dir = Path('augmented_dataset_balanced')
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Created balanced output directory: {output_dir}")
    
    # Configure ImageDataGenerator
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.2, 1.2],
        fill_mode='nearest'
    )
    
    # Organize images by class
    class_images = {}
    image_files = list(input_dir.glob('*.jpg'))
    
    for img_path in image_files:
        # Extract class name from filename (assuming format: UnsharpMask.ID.ClassName.jpg)
        filename_parts = img_path.stem.split('.')
        if len(filename_parts) >= 3:
            class_name = '.'.join(filename_parts[2:])  # Handle multi-part class names
            if class_name not in class_images:
                class_images[class_name] = []
            class_images[class_name].append(img_path)
    
    print(f"Found {len(class_images)} classes:")
    for class_name, images in class_images.items():
        print(f"  {class_name}: {len(images)} images")
    
    # Process each class
    total_processed = 0
    total_augmented = 0
    
    for class_name, images in class_images.items():
        print(f"\nProcessing class: {class_name}")
        
        for img_path in tqdm(images, desc=f"Augmenting {class_name}"):
            try:
                # Load image
                img = load_img(img_path, target_size=None)
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Extract base filename
                base_filename = img_path.stem
                
                # Generate augmented images
                augmentation_count = 0
                
                for i, batch in enumerate(datagen.flow(img_array, batch_size=1)):
                    augmented_img = array_to_img(batch[0])
                    output_filename = f"{base_filename}_aug{i+1}.jpg"
                    output_path = output_dir / output_filename
                    augmented_img.save(output_path, 'JPEG', quality=95)
                    
                    augmentation_count += 1
                    total_augmented += 1
                    
                    if augmentation_count >= 3:  # 3 augmented versions per image
                        break
                
                # Copy original
                original_output_path = output_dir / f"{base_filename}_original.jpg"
                img.save(original_output_path, 'JPEG', quality=95)
                total_processed += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
    
    print(f"\nBalanced augmentation completed!")
    print(f"Total original images: {total_processed}")
    print(f"Total augmented images: {total_augmented}")
    print(f"Total images in balanced dataset: {total_processed + total_augmented}")

if __name__ == "__main__":
    # You can choose which version to run
    print("Choose augmentation method:")
    print("1. Standard augmentation (5 augmented versions per image)")
    print("2. Balanced augmentation (3 augmented versions per image, organized by class)")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "2":
        create_augmented_dataset_with_class_balance()
    else:
        create_augmented_dataset() 