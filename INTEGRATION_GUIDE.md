# Integration Guide: Using the Modular Encoder Framework

This guide explains how to integrate the new modular encoder framework with your existing Med-MoE codebase.

## Overview

The modular encoder framework provides a flexible way to use different image encoders (CLIP, SigCLIP, and potentially others) with your existing models. This allows you to:

1. Easily switch between different encoders with minimal code changes
2. Add new encoders (like MedViT or MedCLIP) in the future
3. Maintain a consistent interface for all encoders

## Steps to Integrate

### 1. Install the Framework

Make sure the encoders package is in your Python path:

```bash
pip install -r encoders/requirements.txt
```

### 2. Modify the Vision Tower Builder

Edit `moellava/model/multimodal_encoder/builder.py` to include our encoder framework:

```python
# Add this import at the top
from encoders import EncoderVisionTower

def build_vision_tower(image_tower, args, **kwargs):
    """Build a vision tower."""
    image_tower_cfg = getattr(args, 'mm_vision_tower_cfg', None)
    
    # First check if we should use the new encoder framework
    if hasattr(args, 'use_encoder_framework') and args.use_encoder_framework:
        print(f"Using encoder framework for {image_tower}")
        return EncoderVisionTower(image_tower, args=args, **kwargs)
    
    # Otherwise use the original logic
    if 'clip-vit' in image_tower.lower():
        from .clip_encoder import CLIPVisionTower
        return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    elif 'siglip' in image_tower.lower():
        from .siglip_encoder import SigLipVisionTower
        return SigLipVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown image tower: {image_tower}')
```

### 3. Update Your Model Configuration

To use the new encoder framework, add the `use_encoder_framework` flag to your model configuration:

```python
# In your model configuration
config = {
    "mm_vision_tower": "openai/clip-vit-large-patch14",
    "use_encoder_framework": True
}
```

### 4. Update Your Training Scripts

Update your training scripts to include the new flag:

```bash
python train.py \
    --vision_tower openai/clip-vit-large-patch14 \
    --use_encoder_framework
```

### 5. Adding New Encoders

To add a new encoder:

1. Create a new file in the `encoders` directory (e.g., `encoders/medclip.py`)
2. Implement the encoder by inheriting from `ImageEncoder`
3. Update `encoders/factory.py` to include your new encoder
4. Update `encoders/__init__.py` to expose your new encoder

### 6. Testing Your Integration

Use the provided example scripts to test the integration:

```bash
# Test basic encoder functionality
python encoder_example.py --encoder clip --image path/to/image.jpg

# Test vision tower bridge
python vision_tower_example.py --model openai/clip-vit-large-patch14 --image path/to/image.jpg
```

## Using Different Encoders

Once integrated, you can switch between encoders simply by changing the model name:

```bash
# Use CLIP
python train.py \
    --vision_tower openai/clip-vit-large-patch14 \
    --use_encoder_framework

# Use SigCLIP
python train.py \
    --vision_tower google/siglip-base-patch16-224 \
    --use_encoder_framework
```

For encoders that don't have automatic detection based on name, you can specify the encoder type explicitly:

```python
# In your code
config = SimpleNamespace(
    mm_vision_tower="custom/model/path",
    use_encoder_framework=True,
    encoder_type="medclip"  # Custom parameter for EncoderVisionTower
)
vision_tower = build_vision_tower(config.mm_vision_tower, config)
```

## Troubleshooting

If you encounter issues:

1. **Encoder not found**: Make sure the encoders package is in your Python path
2. **Model not found**: Check that you're using the correct model name for the encoder
3. **Feature mismatch**: Ensure the EncoderVisionTower's feature selection matches what your model expects

For more help, refer to the documentation in the encoders package. 