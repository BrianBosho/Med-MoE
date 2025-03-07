# Downloading Data and Model Files for Med-MoE

This document provides instructions for downloading the large data and model files that are not included in the Git repository.

## Why these files are excluded

The repository excludes large data and model files to:
- Reduce repository size and clone time
- Prevent version control issues with large binary files
- Follow best practices for ML projects

## Files and directories excluded

The following are excluded from the repository and need to be downloaded separately:

- `data/` - Contains medical image datasets
- `datafiles/` - Contains processed data files
- `cache_dir/` - Contains cached model files
- `MedMoE-phi2/` - Contains the Phi-2 model weights
- `clip-vit-large-patch14-336/` - Contains the CLIP model weights
- Model files (*.pth, *.bin, *.safetensors, etc.)
- Large data files (*.arrow, *.zip, etc.)

## How to download

### Option 1: Use HuggingFace Hub

Many of the model files can be downloaded from HuggingFace. For example:

```bash
# Download CLIP model
python -c "from transformers import CLIPProcessor, CLIPModel; model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336', cache_dir='./clip-vit-large-patch14-336')"

# Download Phi-2 model
python -c "from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2', cache_dir='./MedMoE-phi2')"
```

### Option 2: Download from project storage

If you're part of the project team or have access to the project's storage:

```bash
# Example command to download using rsync (adjust paths as needed)
rsync -av user@remote-server:/path/to/med-moe-data/ ./data/
rsync -av user@remote-server:/path/to/med-moe-models/ ./
```

### Option 3: Use the provided scripts

The project includes scripts for downloading the necessary files:

```bash
# Example: Download images
python download_images.py

# Example: Process and prepare datafiles
python scripts/prepare_data.py
```

## Verification

After downloading, verify that the directory structure matches what's expected:

```
Med-MoE/
├── data/
│   ├── pathvqa_images/
│   └── slake_images/
├── datafiles/
│   ├── path_vqa_test/
│   └── slake_test/
├── MedMoE-phi2/
│   └── [model files]
├── clip-vit-large-patch14-336/
│   └── [model files]
```

## Troubleshooting

If you encounter issues downloading or using the models:

1. Check that you have sufficient disk space (at least 20GB recommended)
2. Ensure you have the correct permissions to download from HuggingFace
3. Contact the project maintainers if you need access to private repositories

For more information, refer to the main README.md file. 