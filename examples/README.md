# Outformer Examples

This directory contains example implementations showcasing different use cases of the Outformer library.

## Examples Overview

### 1. Chain of Thought (`chain_of_thought.py`)
Demonstrates how to implement step-by-step reasoning with structured output. This example shows how to:
- Guide through mathematical problem-solving
- Generate structured output with explanations and intermediate steps
- Format the output in a clear, hierarchical JSON structure

Example output:
```json
{
    "steps": [
        {
            "explanation": "Subtract 7 from both sides to isolate the term with x.",
            "output": "8x = -30"
        },
        {
            "explanation": "Divide both sides by 8 to solve for x.",
            "output": "x = -3.75"
        }
    ],
    "final_answer": "x = -3.75"
}
```

### 2. Function Calling (`function_calling.py`)
Shows how to implement a function calling system that can:
- Detect when a function call is needed
- Choose the appropriate function based on user input
- Extract and validate parameters
- Handle multiple function types (weather, time, calculations)

Available functions:
- `get_weather`: Get current temperature for a location
- `get_time`: Get current time for a timezone
- `calculate`: Perform basic mathematical calculations

### 3. Information Extraction (`information_extraction.py`)
Demonstrates structured information extraction from natural language text. Features:
- Extracts named entities and relationships
- Handles arrays of information
- Validates minimum required items
- Produces clean, structured JSON output

Example output:
```json
{
    "name": "Science Fair",
    "date": "Friday",
    "participants": ["Alice", "Bob"]
}
```

## Running the Examples

To run any example:

1. Make sure you have the required dependencies installed:
```bash
pip install outformer
```

2. Run the desired example:
```bash
python examples/chain_of_thought.py
python examples/function_calling.py
python examples/information_extraction.py
```

Note: The examples use the Qwen/Qwen3-1.7B model by default. You can modify the model in the code to use a different one if needed. 
