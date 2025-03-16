# Enhanced Builder for MoELLaVA

The enhanced builder module provides a modular, flexible architecture for loading and configuring multimodal models in MoELLaVA.

## Key Features

### Modular Architecture
- Split into separate functions for loading configurations, tokenizers, vision encoders, and language models
- Better code organization for easier maintenance and testing
- Cleaner error handling with meaningful messages

### Flexible Vision Encoder Support
- Plugin framework for custom vision encoders (CLIP, SigClip, BiomedCLIP)
- Support for custom image sizes and aspect ratios
- Configurable feature extraction layer and strategy
- Advanced encoder types:
  - **SigClip**: Google's SigLip model with sigmoid loss functions (better for zero-shot tasks)
  - **BiomedCLIP**: Microsoft's biomedical CLIP model (specialized for medical imagery)
- Automatic detection of encoder types based on model names

### Enhanced Configuration System
- Support for both dictionary-based and object-based configuration
- Ability to override settings at load time
- Flexible parameter handling with sane defaults

### Comprehensive Model Support
- Compatible with LLaMA, Mistral, Phi, Qwen, MiniCPM, and StableLM variants
- Support for MoE (Mixture of Experts) model variants
- LoRA fine-tuning support with proper weight merging

## Usage Examples

### Basic Usage
```python
from moellava.model.builder_enhanced import load_pretrained_model

# Load a model with default settings
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda"
)
```

### Customizing Vision Encoder
```python
from dataclasses import dataclass

# Create a custom vision configuration
@dataclass
class VisionConfig:
    mm_vision_tower: str = "openai/clip-vit-large-patch14"
    image_size: int = 336
    vision_feature_layer: int = -2

# Load model with custom vision configuration
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    custom_vision_config=VisionConfig()
)
```

### Using SigClip Encoder
```python
# Using SigClip with automatic detection (based on model path)
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    custom_vision_config=VisionConfig(
        mm_vision_tower="google/siglip-base-patch16-224"
    )
)

# Explicitly specifying SigClip encoder
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    encoder_type="sigclip",
    custom_vision_config=VisionConfig(
        mm_vision_tower="google/siglip-base-patch16-224"
    )
)
```

### Using BiomedCLIP Encoder
```python
# Using BiomedCLIP with automatic detection
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    custom_vision_config=VisionConfig(
        mm_vision_tower="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
)

# Explicitly specifying BiomedCLIP encoder
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path="path/to/model",
    device="cuda",
    encoder_type="biomedclip",
    custom_vision_config=VisionConfig(
        mm_vision_tower="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
)
```

### Using Dictionary Configuration
```python
# Configure with a dictionary instead of objects
config = {
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "mm_vision_tower": "openai/clip-vit-large-patch14",
    "mm_hidden_size": 768
}

model, image_processor, tokenizer, context_len = load_pretrained_model(
    config,
    device="cuda"
)
```

## Testing
Several test scripts are included to verify functionality:
- `test_builder_enhanced.py`: Tests basic functionality
- `test_small_model.py`: Tests loading a small model end-to-end
- `test_sigclip_encoder.py`: Tests the SigClip encoder integration
- `test_biomedclip_encoder.py`: Tests the BiomedCLIP encoder integration

Run the tests with:
```bash
PYTHONPATH=/path/to/Med-MoE python test_builder_enhanced.py
```

## Custom Encoder Framework
To implement your own custom encoder:
1. Create a new file in the `encoders` folder (e.g., `my_encoder.py`)
2. Implement a class with `load_model()`, `to()`, and `forward()` methods
3. Update `encoders/__init__.py` to expose your encoder
4. Update `encoders/factory.py` to support your encoder type

## Future Improvements
- Add support for more vision encoder types
- Improve handling of quantized models
- Add batched processing support for more efficient inference 