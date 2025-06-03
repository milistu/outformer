# Quick Start

Here's a simple example to get you started with Outformer:

## Basic Usage

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
        "gender": {
            "type": "string",
            "enum": ["Female", "Male", "Unisex"],
        },
    },
}

# Your input prompt
prompt = """
Extract key information from the product description:
adidas Men's Powerlift.3 Cross-Trainer Shoes
"""

# Generate structured output
generated_data = jsonformer.generate(schema=json_schema, prompt=prompt)

# Highlight generated values  
highlight_values(generated_data)
```

### Expected Output
The code above will generate a structured JSON output:
```json
{
    "brand": "Adidas",
    "model": "Powerlift.3", 
    "gender": "Male"
}
```
When using `highlight_values()`, the generated values will be highlighted in color in your terminal.