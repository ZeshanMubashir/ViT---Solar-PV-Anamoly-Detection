# Image Unsharp Mask Filter

This script applies an unsharp mask filter to all images in your dataset and saves them with a specific naming convention.

## Prerequisites

Make sure you have Python installed. Then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your images are in the `images/` directory
2. Ensure your metadata file `module_metadata.json` is in the root directory
3. Run the script:

```bash
python apply_unsharp_filter.py
```

## Filter Settings

The script applies the following unsharp mask settings as specified:
- **Blur Radius**: 2 pixels
- **Unsharp Strength**: 150% (1.5)

## Output

- Filtered images will be saved in a new `filtered_dataset/` directory
- File naming convention: `UnsharpMask.{imagename}.{classname}.jpg`
  - `{imagename}`: Original image ID (0, 1, 2, etc.)
  - `{classname}`: Anomaly class from metadata (e.g., "No-Anomaly", "Cell", "Hot-Spot", etc.)

## Example Output Files

```
filtered_dataset/
├── UnsharpMask.0.Offline-Module.jpg
├── UnsharpMask.1.Cell.jpg
├── UnsharpMask.2.Hot-Spot.jpg
└── ...
```

The script will also display progress updates and a summary of the class distribution when complete. 