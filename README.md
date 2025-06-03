![Outformer Logo](./docs/img/logo_dark.png#gh-dark-mode-only "Outformer Dark Logo")
![Outformer Logo](./docs/img/logo_light.png#gh-light-mode-only "Outformer Light Logo")

# Outformer: Structure Outputs from Language Models

[![Python Version](https://img.shields.io/badge/python-3.10%2B-8c52ff.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/outformer.svg?color=5ce1e6)](https://badge.fury.io/py/outformer)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?color=393be5)](LICENSE)

Outformer is a powerful library that enables language models to generate structured outputs. It ensures always valid JSON outputs by generating only values while maintaining the structural integrity of your schema.

## Features

- 🔄 **Structured Output Generation**: Generate valid JSON outputs from language models
- 🎯 **Schema Validation**: Ensure outputs conform to your JSON schema
- 🛠️ **Flexible Integration**: Works with any Hugging Face transformer model
- 🚀 **Easy to Use**: Simple API with minimal configuration
- 🎨 **Value Highlighting**: Visualize generated values in your JSON structure

## Installation

We recommend **Python 3.10+, [PyTorch 2.7.0+](https://pytorch.org/get-started/locally/), [transformers](https://github.com/huggingface/transformers) v4.51.3+**.

### Install via pip
```bash
pip install outformer
```

### Install from source
```bash
git clone https://github.com/milistu/outformer.git
cd outformer
pip install -e .
```

## Quick Start

Here's a simple example to get you started:

![Quick Start Example](./docs/img/quick_start_dark.png#gh-dark-mode-only)
![Quick Start Example](./docs/img/quick_start_light.png#gh-light-mode-only)

<details>
<summary>Click to expand code example</summary>

```python
from outformer import Jsonformer, highlight_values
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize model and tokenizer
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create Jsonformer instance
jsonformer = Jsonformer(model, tokenizer, max_tokens_string=100)

# Define your JSON schema
json_schema = {
    "type": "object",
    "properties": {
        "brand": {
            "type": "string",
            "description": "Brand of the product",
        },
        "model": {
            "type": "string",
            "description": "Model of the product",
        },
        "product_type": {
            "type": "string",
            "description": "Type of the product",
        },
        "gender": {
            "type": "string",
            "enum": ["Female", "Male", "Unisex"],
        },
        "color": {
            "type": "string",
            "description": "Color of the product",
        },
        "features": {
            "type": "array",
            "minItems": 3,
            "items": {
                "type": "string",
                "description": "Features of the product that may be relevant for the customer",
            },
        },
    },
}

# Your input prompt
prompt = """
Extract key information from the product description:

adidas Men's Powerlift.3 Cross-Trainer Shoes
A powerful shoe with lockdown fit. Made with an extra-wide design that allows the foot to spread, these men's lifting/weight-training shoes pair a snug-fitting upper with a wide midfoot strap for extra support. A high-density die-cut wedge midsole keeps you close to the ground.
100% Synthetic leather
Imported
Rubber sole
Removable Insole
"""

# Generate structured output
generated_data = jsonformer.generate(schema=json_schema, prompt=prompt)

# Highlight generated values
highlight_values(generated_data)
```

The code above will generate a structured JSON output and display it with highlighted values. Here's what you'll get:

```json
{
    "brand": "Adidas",
    "model": "Powerlift.3",
    "product_type": "Lifting/Weight Training Shoes",
    "gender": "Male",
    "color": "Black",
    "features": [
        "Extra wide design for optimal foot support",
        "High-density die-cut wedge midsole",
        "Rubber sole with removable insole",
        "100% synthetic leather"
    ]
}
```

When using `highlight_values()`, the output will be displayed in your terminal with the generated values highlighted in color, making it easy to distinguish between the structure and the generated content.
</details>

## Advanced Usage

### Configuration Options

The `Jsonformer` class accepts several configuration parameters:

- `debug` (bool): Enable debug mode for detailed generation process
- `max_array_length` (int): Maximum number of elements in an array
- `max_tokens_number` (int): Maximum number of tokens for number generation
- `max_tokens_string` (int): Maximum number of tokens for string generation
- `temperature` (float): Sampling temperature for generation
- `generation_marker` (str): Marker for tracking generation position
- `max_attempts` (int): Maximum attempts for value generation

### Supported JSON Schema Features

- Basic types: string, number, boolean
- Arrays with min/max items
- Objects with nested properties
- Enums for constrained string values
- Descriptions for better generation context

## Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citing & Authors

The idea for this repository was inspired by [jsonformer](https://github.com/1rgs/jsonformer).

Maintainer: [Milutin Studen](https://github.com/milistu)

## Support

If you encounter any issues or have questions, please open an issue on our [GitHub repository](https://github.com/milistu/outformer/issues).

