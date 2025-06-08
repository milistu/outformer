from typing import Any, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer

from outformer import Jsonformer, highlight_values


def format_functions(functions: Dict[str, Any]) -> str:
    formatted = []
    for name, info in functions.items():
        params = []
        if "properties" in info["parameters"]:
            for param_name, param_info in info["parameters"]["properties"].items():
                params.append(f"{param_name}: {param_info['description']}")

        param_str = ", ".join(params) if params else "No parameters"
        formatted.append(f" - {name}: {param_str}: {info['description']}")

    return "\n".join(formatted)


def execute_function_call(function_call: Dict[str, Any]) -> None:
    """Simulate function execution based on the generated call."""
    print("FUNCTION CALL ANALYSIS:")
    requires_function_call = function_call.get("requires_function_call", False)
    print(f"Requires function call: {requires_function_call}")

    if not requires_function_call:
        print("No function call required, This is a general conversation.")
        return

    function_name = function_call.get("function_name")
    parameters = function_call.get("parameters", {})
    reasoning = function_call.get("reasoning", "No reasoning provided")

    print(f"Function: {function_name}")
    print(f"Parameters: {parameters}")
    print(f"Reasoning: {reasoning}")

    # Simulate function execution
    if function_name == "get_weather":
        location = parameters.get("location", "Unknown")
        print(f"🌤️  get_weather(location='{location}')")

    elif function_name == "get_time":
        timezone = parameters.get("timezone", "Unknown")
        print(f"🕐 get_time(timezone='{timezone}')")

    elif function_name == "calculate":
        expression = parameters.get("expression", "No expression")
        print(f"🧮 calculate(expression='{expression}')")

    else:
        print(f"Unknown function: {function_name}")


def main():
    model_name = "Qwen/Qwen3-1.7B"
    cache_dir = ".cache"

    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        "get_time": {
            "description": "Get current time for a given timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone e.g. America/New_York",
                    }
                },
            },
        },
        "calculate": {
            "description": "Perform basic mathematical calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate e.g. '2 + 3 * 4'",
                    }
                },
            },
        },
    }

    # Schema for initial function detection
    function_detection_schema = {
        "type": "object",
        "properties": {
            "requires_function_call": {
                "type": "boolean",
                "description": "Whether the user request requires calling a function",
            },
            "function_name": {
                "type": "string",
                "enum": list(available_functions.keys()) + ["none"],
                "description": "Name of the function to call, or 'none' if no function needed",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why this function was chosen",
            },
        },
    }

    # Function-specific parameter schemas
    parameter_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country divided by comma. e.g. City, Country",
                }
            },
        },
        "get_time": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone in standard format e.g. 'UTC±HH:MM' or 'Region/City' format",
                }
            },
        },
        "calculate": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate e.g. 'X + Y'",
                }
            },
        },
    }

    # Test different user requests
    test_requests = [
        "What is the weather like in Paris today?",
        "What time is it in New York?",
        "Calculate 15 * 42 + 7",
        "Hello, how are you?",  # This shouldn't require a function call
    ]

    former = Jsonformer(model, tokenizer, max_tokens_string=100)

    for request in test_requests:
        print(f"\n{'='*60}")
        print(f"USER REQUEST: {request}")
        print("=" * 60)

        # Stage 1: Determine if function call is needed and which function
        detection_prompt = f"""
        Available functions:
        {format_functions(available_functions)}

        User request: {request}

        Analyze the user request and determine:
        1. Does it require calling a function?
        2. If yes, which function should be called?
        3. Provide reasoning for your choice.
        """

        try:
            # Stage 1: Function detection
            function_detection = former.generate(
                function_detection_schema, detection_prompt
            )

            print("\nFUNCTION DETECTION:")
            highlight_values(function_detection)

            # Stage 2: Generate parameters only if function call is needed
            if (
                function_detection.get("requires_function_call")
                and function_detection.get("function_name") != "none"
            ):
                function_name = function_detection["function_name"]
                parameter_schema = parameter_schemas.get(function_name)

                if parameter_schema:
                    param_prompt = f"""
                    User request: {request}
                    Selected function: {function_name}
                    Function description: {available_functions[function_name]['description']}

                    Extract the required parameters for the {function_name} function from the user request.
                    """

                    parameters = former.generate(parameter_schema, param_prompt)

                    print(f"\nGENERATED PARAMETERS FOR {function_name.upper()}:")
                    highlight_values(parameters)

                    # Combine results
                    complete_function_call = {
                        **function_detection,
                        "parameters": parameters,
                    }

                    # Simulate function execution
                    execute_function_call(complete_function_call)
                else:
                    print(f"No parameter schema found for function: {function_name}")
            else:
                print("No function call required - this is general conversation.")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
