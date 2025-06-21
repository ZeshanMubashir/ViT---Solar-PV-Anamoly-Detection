import json
import os
from PIL import Image, ImageFilter, ImageEnhance
import shutil
from pathlib import Path

def apply_unsharp_mask(image, blur_radius=2, unsharp_strength=1.5):
    """
    Apply unsharp mask filter to an image
    
    Args:
        image: PIL Image object
        blur_radius: Blur radius for the unsharp mask (default: 2)
        unsharp_strength: Strength of the unsharp effect (default: 1.5 for 150%)
    
    Returns:
        PIL Image object with unsharp mask applied
    """
    # Create a blurred version of the image
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Apply the unsharp mask: original + (original - blurred) * strength
    # This is the standard unsharp mask formula
    from PIL import ImageChops
    
    # Calculate the difference (high-pass filter)
    mask = ImageChops.subtract(image, blurred)
    
    # Scale the mask by the strength factor
    mask = ImageEnhance.Brightness(mask).enhance(unsharp_strength)
    
    # Add the scaled mask to the original image
    result = ImageChops.add(image, mask)
    
    return result

def create_filtered_dataset():
    """
    Main function to create filtered dataset with unsharp mask applied
    """
    
    # Load metadata
    print("Loading metadata...")
    with open('module_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create output directory
    output_dir = Path('filtered_dataset')
    output_dir.mkdir(exist_ok=True)
    
    print(f"Created output directory: {output_dir}")
    
    # Process each image
    total_images = len(metadata)
    processed = 0
    
    print(f"Processing {total_images} images...")
    
    for image_id, image_info in metadata.items():
        try:
            # Get image path and class name
            image_path = image_info['image_filepath']
            class_name = image_info['anomaly_class']
            
            # Check if image file exists
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, skipping...")
                continue
            
            # Load the image
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (in case of grayscale or RGBA)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply unsharp mask filter
                # Blur Radius = 2, Unsharp Strength = 150% (1.5)
                filtered_img = apply_unsharp_mask(img, blur_radius=2, unsharp_strength=1.5)
                
                # Create output filename: UnsharpMask.{imagename}.{classname}
                # Image name is extracted from the image_id
                output_filename = f"UnsharpMask.{image_id}.{class_name}.jpg"
                output_path = output_dir / output_filename
                
                # Save the filtered image
                filtered_img.save(output_path, 'JPEG', quality=95)
                
                processed += 1
                if processed % 50 == 0:  # Progress update every 50 images
                    print(f"Processed {processed}/{total_images} images...")
                    
        except Exception as e:
            print(f"Error processing image {image_id}: {str(e)}")
            continue
    
    print(f"\nCompleted! Processed {processed} out of {total_images} images.")
    print(f"Filtered images saved in: {output_dir}")
    
    # Print summary statistics
    class_counts = {}
    for image_info in metadata.values():
        class_name = image_info['anomaly_class']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\nClass distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} images")

if __name__ == "__main__":
    create_filtered_dataset() 