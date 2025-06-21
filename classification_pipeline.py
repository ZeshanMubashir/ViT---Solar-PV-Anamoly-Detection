import json
import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance
from collections import defaultdict, Counter

class ClassificationDatasetPipeline:
    """
    Pipeline to create datasets for different classification tasks:
    1. 2-Class: No-Anomaly vs Anomaly (40,000 images)
    2. 11-Class: Anomaly Types Only (23,100 images) 
    3. 12-Class: No-Anomaly + 11 Anomaly Types (25,200 images)
    """
    
    def __init__(self, source_dir='augmented_dataset', metadata_file='module_metadata.json'):
        self.source_dir = Path(source_dir)
        self.metadata_file = metadata_file
        self.metadata = self._load_metadata()
        self.anomaly_classes = [
            'Cell', 'Cell-Multi', 'Cracking', 'Diode', 'Diode-Multi', 
            'Hot-Spot', 'Hot-Spot-Multi', 'Offline-Module', 
            'Shadowing', 'Soiling', 'Vegetation'
        ]
        
    def _load_metadata(self):
        """Load and return metadata"""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def _get_class_from_filename(self, filename):
        """Extract class name from augmented filename"""
        # Format: UnsharpMask.ID.ClassName_aug_type.jpg or UnsharpMask.ID.ClassName_original.jpg
        if filename.startswith('UnsharpMask.'):
            # Remove extension
            name_without_ext = filename.replace('.jpg', '')
            
            # Split by underscore to separate the augmentation part
            if '_' in name_without_ext:
                # Get the part before the first underscore after the class name
                # Find where the class name ends by looking for _aug or _original
                if '_aug_' in name_without_ext:
                    base_part = name_without_ext.split('_aug_')[0]
                elif '_original' in name_without_ext:
                    base_part = name_without_ext.split('_original')[0]
                else:
                    base_part = '_'.join(name_without_ext.split('_')[:-1])
            else:
                base_part = name_without_ext
            
            # Now extract class name from base_part
            # Format should be: UnsharpMask.ID.ClassName
            parts = base_part.split('.')
            if len(parts) >= 3:
                # Join everything after the ID (in case class name has dots)
                class_name = '.'.join(parts[2:])
                return class_name
        return None
    
    def _sample_balanced_images(self, class_distribution, target_per_class):
        """Sample images to achieve balanced distribution"""
        selected_files = []
        
        for class_name, files in class_distribution.items():
            if len(files) >= target_per_class:
                # Randomly sample target_per_class images
                selected = random.sample(files, target_per_class)
            else:
                # Use all available and repeat some if needed
                selected = files * (target_per_class // len(files))
                remaining = target_per_class % len(files)
                if remaining > 0:
                    selected.extend(random.sample(files, remaining))
            
            selected_files.extend(selected)
        
        return selected_files
    
    def create_2class_dataset(self):
        """
        Create 2-Class Dataset: No-Anomaly vs Anomaly
        Target: 40,000 images (20,000 per class)
        """
        print("🔄 Creating 2-Class Dataset (No-Anomaly vs Anomaly)")
        
        output_dir = Path('datasets/2_class_dataset')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class directories
        no_anomaly_dir = output_dir / 'No-Anomaly'
        anomaly_dir = output_dir / 'Anomaly'
        no_anomaly_dir.mkdir(exist_ok=True)
        anomaly_dir.mkdir(exist_ok=True)
        
        # Organize files by binary classification
        no_anomaly_files = []
        anomaly_files = []
        
        for img_file in self.source_dir.glob('*.jpg'):
            class_name = self._get_class_from_filename(img_file.name)
            if class_name == 'No-Anomaly':
                no_anomaly_files.append(img_file)
            elif class_name in self.anomaly_classes:
                anomaly_files.append(img_file)
        
        print(f"Found {len(no_anomaly_files)} No-Anomaly images")
        print(f"Found {len(anomaly_files)} Anomaly images")
        
        # Sample 20,000 from each class
        selected_no_anomaly = random.sample(no_anomaly_files, min(20000, len(no_anomaly_files)))
        selected_anomaly = random.sample(anomaly_files, min(20000, len(anomaly_files)))
        
        # Copy files
        for img_file in selected_no_anomaly:
            shutil.copy2(img_file, no_anomaly_dir / img_file.name)
        
        for img_file in selected_anomaly:
            shutil.copy2(img_file, anomaly_dir / img_file.name)
        
        print(f"✅ 2-Class Dataset Created:")
        print(f"   📁 No-Anomaly: {len(selected_no_anomaly)} images")
        print(f"   📁 Anomaly: {len(selected_anomaly)} images")
        print(f"   📁 Total: {len(selected_no_anomaly) + len(selected_anomaly)} images")
        print(f"   💾 Saved to: {output_dir}")
    
    def create_11class_dataset(self):
        """
        Create 11-Class Dataset: Anomaly Types Only
        Target: 23,100 images (2,100 per class)
        """
        print("\n🔄 Creating 11-Class Dataset (Anomaly Types Only)")
        
        output_dir = Path('datasets/11_class_dataset')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Organize files by anomaly class
        class_files = defaultdict(list)
        
        for img_file in self.source_dir.glob('*.jpg'):
            class_name = self._get_class_from_filename(img_file.name)
            if class_name in self.anomaly_classes:
                class_files[class_name].append(img_file)
        
        # Create directories and sample balanced data
        total_copied = 0
        for class_name in self.anomaly_classes:
            class_dir = output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            available_files = class_files[class_name]
            target_count = 2100
            
            if len(available_files) >= target_count:
                selected_files = random.sample(available_files, target_count)
            else:
                # Use all available and augment by repeating
                selected_files = available_files * (target_count // len(available_files))
                remaining = target_count % len(available_files)
                if remaining > 0:
                    selected_files.extend(random.sample(available_files, remaining))
            
            # Copy files
            for img_file in selected_files:
                shutil.copy2(img_file, class_dir / img_file.name)
            
            total_copied += len(selected_files)
            print(f"   📁 {class_name}: {len(selected_files)} images (from {len(available_files)} available)")
        
        print(f"✅ 11-Class Dataset Created:")
        print(f"   📊 Total: {total_copied} images")
        print(f"   💾 Saved to: {output_dir}")
    
    def create_12class_dataset(self):
        """
        Create 12-Class Dataset: No-Anomaly + 11 Anomaly Types
        Target: 25,200 images (2,100 per class)
        """
        print("\n🔄 Creating 12-Class Dataset (No-Anomaly + 11 Anomaly Types)")
        
        output_dir = Path('datasets/12_class_dataset')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Include No-Anomaly class
        all_classes = ['No-Anomaly'] + self.anomaly_classes
        
        # Organize files by class
        class_files = defaultdict(list)
        
        for img_file in self.source_dir.glob('*.jpg'):
            class_name = self._get_class_from_filename(img_file.name)
            if class_name in all_classes:
                class_files[class_name].append(img_file)
        
        # Create directories and sample balanced data
        total_copied = 0
        for class_name in all_classes:
            class_dir = output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            available_files = class_files[class_name]
            target_count = 2100
            
            if len(available_files) >= target_count:
                selected_files = random.sample(available_files, target_count)
            else:
                # Use all available and augment by repeating
                selected_files = available_files * (target_count // len(available_files))
                remaining = target_count % len(available_files)
                if remaining > 0:
                    selected_files.extend(random.sample(available_files, remaining))
            
            # Copy files
            for img_file in selected_files:
                shutil.copy2(img_file, class_dir / img_file.name)
            
            total_copied += len(selected_files)
            print(f"   📁 {class_name}: {len(selected_files)} images (from {len(available_files)} available)")
        
        print(f"✅ 12-Class Dataset Created:")
        print(f"   📊 Total: {total_copied} images")
        print(f"   💾 Saved to: {output_dir}")
    
    def create_summary_report(self):
        """Create a summary report of all datasets"""
        print("\n📋 DATASET PIPELINE SUMMARY REPORT")
        print("="*60)
        
        datasets = [
            ('2_class_dataset', '2-Class (No-Anomaly vs Anomaly)'),
            ('11_class_dataset', '11-Class (Anomaly Types Only)'),
            ('12_class_dataset', '12-Class (All Classes)')
        ]
        
        for dataset_name, description in datasets:
            dataset_path = Path(f'datasets/{dataset_name}')
            if dataset_path.exists():
                print(f"\n📁 {description}")
                print(f"   Path: {dataset_path}")
                
                total_images = 0
                class_counts = {}
                
                for class_dir in dataset_path.iterdir():
                    if class_dir.is_dir():
                        count = len(list(class_dir.glob('*.jpg')))
                        class_counts[class_dir.name] = count
                        total_images += count
                
                print(f"   Total Images: {total_images}")
                print(f"   Classes: {len(class_counts)}")
                
                for class_name, count in sorted(class_counts.items()):
                    print(f"     • {class_name}: {count} images")
                
                # Calculate size
                try:
                    size_mb = sum(f.stat().st_size for f in dataset_path.rglob('*.jpg')) / (1024 * 1024)
                    print(f"   Size: {size_mb:.2f} MB")
                except:
                    print(f"   Size: Unable to calculate")
        
        print("\n" + "="*60)
        print("✅ All datasets created successfully!")
        print("📝 Ready for classification model training")
    
    def run_full_pipeline(self):
        """Run the complete pipeline for all classification tasks"""
        print("🚀 Starting Classification Dataset Pipeline")
        print("="*60)
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Create all datasets
        self.create_2class_dataset()
        self.create_11class_dataset()
        self.create_12class_dataset()
        
        # Generate summary report
        self.create_summary_report()

def main():
    """Main function to run the pipeline"""
    
    # Check if source directory exists
    if not Path('augmented_dataset').exists():
        print("❌ Error: 'augmented_dataset' directory not found!")
        print("Please run the augmentation script first.")
        return
    
    if not Path('module_metadata.json').exists():
        print("❌ Error: 'module_metadata.json' file not found!")
        return
    
    # Initialize and run pipeline
    pipeline = ClassificationDatasetPipeline()
    pipeline.run_full_pipeline()

if __name__ == "__main__":
    main() 