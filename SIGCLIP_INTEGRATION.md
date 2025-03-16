# SigClip Integration Guide

This document explains how to use Google's SigLip (Sigmoid Loss) vision-language models as encoders in the enhanced MoELLaVA builder.

## Overview

SigLip (also referred to as SigClip in our implementation) is Google's vision-language model that uses sigmoid loss instead of the standard contrastive loss used in CLIP. SigLip models have shown excellent performance on vision-language tasks and can be used as a drop-in replacement for CLIP encoders in multimodal applications.

## Requirements

- Transformers library (version 4.34.0 or higher)
- Hugging Face Hub access for downloading pre-trained models
- MedVQA Enhanced Builder implementation

## Available SigLip Models

Google has released several SigLip models on Hugging Face Hub:

- `google/siglip-base-patch16-224`: Base model with 224x224 input size
- `google/siglip-large-patch16-224`: Large model with 224x224 input size
- `google/siglip-base-patch16-256`: Base model with 256x256 input size
- `google/siglip-large-patch16-256`: Large model with 256x256 input size

## Integration Components

The SigClip integration consists of the following components:

1. **SigClipEncoder Class**: A custom encoder implementation (`encoders/sigclip.py`) that handles loading, device placement, and image encoding using SigLip models.

2. **Factory Update**: The `create_encoder` function in `encoders/factory.py` was updated to support SigClip encoders.

3. **Builder Enhancement**: The `load_vision_encoder` function in `builder_enhanced.py` was updated to automatically detect SigLip models and use the appropriate image processor.

## Usage

### Basic Usage

```python
from moellava.model.builder_enhanced import load_pretrained_model

# Using SigClip with a dictionary configuration
model_config = {
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mm_vision_tower": "google/siglip-base-patch16-224",
    "mm_hidden_size": 768
}

# Load the model with SigClip encoder
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_config,
    encoder_type='sigclip',
)
```

### Automatic Encoder Type Detection

The builder will automatically detect SigLip models based on the model path containing "siglip":

```python
model_config = {
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mm_vision_tower": "google/siglip-base-patch16-224",
    "mm_hidden_size": 768
}

# No need to specify encoder_type - it will be detected automatically
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_config
)
```

### Custom Image Size

You can specify a custom image size using the `custom_vision_config`:

```python
from dataclasses import dataclass

# Define custom configuration
@dataclass
class CustomVisionConfig:
    image_size: int = 224
    vision_feature_layer: int = -2
    vision_feature_select_strategy: str = "default"

# Use the custom configuration
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_config,
    encoder_type='sigclip',
    custom_vision_config=CustomVisionConfig()
)
```

## Testing

You can verify the SigClip integration using the provided test script:

```bash
PYTHONPATH=/path/to/Med-MoE python test_sigclip_encoder.py
```

The test script checks both the direct loading of the SigClip encoder and the integration with a full model.

## Troubleshooting

- If you encounter the error `ModuleNotFoundError: No module named 'encoders'`, make sure your PYTHONPATH includes the root directory of the project.
- If you see warnings about missing processor configurations, these are normal and occur because SigLip image processors are relatively new in the Transformers library.
- If the SigClip encoder fails to load, verify that your Transformers library is updated to a version that supports SigLip models.

## Performance Considerations

- SigLip models generally perform better on zero-shot classification tasks compared to equivalent CLIP models.
- SigLip may have different compute requirements than CLIP, so consider this when deploying models with SigLip encoders.
- For best performance, use the image size that the model was trained on (specified in the model name, e.g., 224 or 256).

## References

- [SigLip Paper: Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)
- [SigLip Models on Hugging Face Hub](https://huggingface.co/google/siglip-base-patch16-224)
- [Transformers Documentation for SigLip](https://huggingface.co/docs/transformers/model_doc/siglip) 