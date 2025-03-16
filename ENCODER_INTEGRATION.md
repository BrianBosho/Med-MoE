# Using Custom Encoders with MoELLaVA

This document explains how to use custom encoders (like SigClip) with the MoELLaVA codebase.

## Super Simple Usage

To switch between encoders, run the simple demo script:

```bash
# Use CLIP encoder (default)
python simple_encoder_demo.py --model_path path/to/model --image path/to/image.jpg --encoder clip

# Use SigClip encoder
python simple_encoder_demo.py --model_path path/to/model --image path/to/image.jpg --encoder sigclip
```

That's it! Everything else is handled automatically.

## Quick Start

### Simple Encoder Selection Interface

The simplest way to specify which encoder to use is with our encoder selection interface:

```python
from encoders.selector import load_model_with_encoder

# Load a model with the SigClip encoder
tokenizer, model, processor, context_len = load_model_with_encoder(
    model_path="your_model_path",
    encoder="sigclip",  # Choose from: "clip", "sigclip", "medclip", etc.
    encoder_path="google/siglip-base-patch16-224"  # Optional override
)

# Now use the model as normal with your chosen encoder
```

### Demo Script

We also provide a complete demo script that shows how to use different encoders:

```bash
python demo_encoder_selection.py \
  --model_path your_model_path \
  --encoder sigclip \
  --encoder_path google/siglip-base-patch16-224 \
  --image path/to/your/image.jpg
```

### Advanced Usage

For more advanced usage, you can also use the lower-level API:

```python
from moellava.model.builder import load_pretrained_model

# Load the model with SigClip encoder
tokenizer, model, processor, context_len = load_pretrained_model(
    model_path="your_model_path",
    model_base=None,
    model_name="llava_model",
    encoder_type="sigclip"  # Specify which encoder to use
)
```

## Supported Encoder Types

The following encoder types are currently supported:

- `clip` - OpenAI CLIP encoder (default)
- `sigclip` - Google SigLip models
- `medclip` - Medical CLIP models (if implemented)

## Image Size Handling

Different encoders expect different image sizes:
- CLIP models: 224x224 or 336x336
- SigLip models: 224x224 or 384x384

The encoder selection interface automatically handles these differences by:
1. Identifying the proper image size based on the encoder model
2. Configuring the image processor to use the correct size
3. Preventing unnecessary resizing warnings

## Adding New Encoder Types

To add a new encoder type:

1. Implement the encoder in the `encoders` package following the pattern of existing encoders
2. Add it to the `ENCODER_TYPES` dictionary in `encoders/selector.py`
3. Add its image size to the `ENCODER_IMAGE_SIZES` dictionary if it differs from standard sizes

## How It Works

The integration is minimal and only modifies the model loading process in `moellava/model/builder.py`. When you specify an `encoder_type`, the system will:

1. Try to load the specified encoder from the `encoders` framework
2. If successful, it will use that encoder with the `EncoderVisionTower` bridge
3. Configure the correct image size based on the encoder model
4. If the encoder framework is not available, it will fall back to the default CLIP encoder

This approach requires no changes to your model architecture or training code. 