# MoELLaVA Test Files and Documentation

This directory contains test files and documentation for the enhanced MoELLaVA builder and custom encoders.

## Test Files

| File | Description |
|------|-------------|
| `test_biomedclip_encoder.py` | Tests loading and using the BiomedCLIP encoder with models |
| `test_sigclip_encoder.py` | Tests loading and using the SigClip encoder with models |
| `test_builder_enhanced.py` | Tests the enhanced model builder functionality |
| `test_small_model.py` | Tests loading a small model with various configurations |
| `test_image_encoder.py` | Tests the generic image encoder functionality |
| `test_image_encoder_simplified.py` | Simplified version of the image encoder test |
| `test_encoder_config.py` | Tests encoder configuration handling |
| `test_loading.py` | Tests basic model loading functionality |
| `test_sigclip_simple.py` | Simple test for SigClip encoder |
| `test_notebook.py` | Notebook-style test examples |

## Running Tests

To run a test file, use the following command:

```bash
# Make sure to set PYTHONPATH to include the project root
PYTHONPATH=/path/to/Med-MoE python tests/test_file_name.py
```

For example:

```bash
PYTHONPATH=/path/to/Med-MoE python tests/test_biomedclip_encoder.py
```

## Documentation

The `docs` subdirectory contains documentation files:

| File | Description |
|------|-------------|
| `ENHANCED_BUILDER_DOCS.md` | **Comprehensive documentation** for the enhanced builder, including all encoder types and projectors |
| `BIOMEDCLIP_INTEGRATION.md` | Specific details about BiomedCLIP integration |
| `README_ENHANCED_BUILDER.md` | Original documentation for the enhanced builder |

The main documentation file is `ENHANCED_BUILDER_DOCS.md`, which covers all aspects of the enhanced builder and custom encoders.

## Development

When creating new tests, please follow these guidelines:

1. Use clear naming conventions (`test_feature_name.py`)
2. Include docstrings explaining the purpose of the test
3. Add proper error handling and informative messages
4. Add your test to this README file

## Examples

See the individual test files for examples of how to use different features of the MoELLaVA framework. 