# Examples

This section provides practical examples of using Outformer for different use cases. For the complete, runnable examples, visit our [examples directory](https://github.com/milistu/outformer/tree/main/examples) in the repository.

> Note: This is a growing collection of examples. We welcome contributions! If you have an interesting use case, feel free to submit a pull request.

## Chain of Thought

The chain of thought example demonstrates how to implement step-by-step reasoning with structured output. Full code available in [chain_of_thought.py](https://github.com/milistu/outformer/blob/main/examples/chain_of_thought.py).

Here's a simplified version of the implementation:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from outformer import Jsonformer, highlight_values

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# Define schema for step-by-step reasoning
schema = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "output": {"type": "string"},
                },
            },
        },
        "final_answer": {"type": "string"},
    },
}

# Create Jsonformer instance
former = Jsonformer(model, tokenizer, max_tokens_string=100)

# Generate structured output
math_reasoning = former.generate(schema, """
You are a helpful math tutor. Guide the user through the solution step by step.
how can I solve 8x + 7 = -23
""")

# Highlight the generated values
highlight_values(math_reasoning)
```

## Function Calling

This example shows how to implement a function calling system that can detect when a function is needed and extract parameters. Full code available in [function_calling.py](https://github.com/milistu/outformer/blob/main/examples/function_calling.py).

Here's a simplified version of the implementation:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from outformer import Jsonformer, highlight_values

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# Define available functions
available_functions = {
    "get_weather": {
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia",
                },
            },
        },
    },
    # ... other functions
}

# Create Jsonformer instance
former = Jsonformer(model, tokenizer)

# Generate function call
function_call = former.generate(function_detection_schema, """
Available functions:
{format_functions(available_functions)}

User request: What is the weather like in Paris today?
""")
```

## Information Extraction

This example demonstrates how to extract structured information from natural language text. Full code available in [information_extraction.py](https://github.com/milistu/outformer/blob/main/examples/information_extraction.py).

Here's a simplified version of the implementation:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from outformer import Jsonformer, highlight_values

# Initialize model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# Define schema for information extraction
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "The name of the event"},
        "date": {"type": "string", "description": "The date of the event"},
        "participants": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "string",
                "description": "The name of the participant",
            },
        },
    },
}

# Create Jsonformer instance
former = Jsonformer(model, tokenizer)

# Generate structured output
event = former.generate(schema, """
Extract the event information.
Alice and Bob are going to a science fair on Friday.
""")

# Highlight the generated values
highlight_values(event)
```

## Contributing Examples

We welcome contributions to our examples collection! If you have an interesting use case or implementation that demonstrates Outformer's capabilities, please submit a pull request.
Your contribution will help others learn and make better use of Outformer. 