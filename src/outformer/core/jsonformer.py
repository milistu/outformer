import json
from typing import Any, Dict, List, Optional, Union

import torch
from termcolor import cprint
from transformers import PreTrainedModel, PreTrainedTokenizer

from outformer.core.token_processors import (
    NumberStoppingCriteria,
    OutputCommaAndBracketTokens,
    OutputNumbersTokens,
    StringStoppingCriteria,
)


class Jsonformer:
    """
    A class that generates structured JSON outputs from language models.

    1. Only generates content values, not structural elements
    2. Follows the provided JSON schema
    3. Builds the JSON object incrementally
    4. Uses a token processor to stop generation at the appropriate time

    This ensures that the output is always a valid JSON object conforming to the specified schema.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        schema: Dict[str, Any],
        prompt: str,
        *,
        debug: bool = False,
        max_array_length: int = 10,
        max_tokens_number: int = 6,
        max_tokens_string: int = 10,
        temperature: float = 0.7,
        generation_marker: str = "|GENERATION|",
        num_retries: int = 3,
    ) -> None:
        """
        Initialize a Jsonformer instance.

        Args:
            model (PreTrainedModel): The model to use for generation.
            tokenizer (PreTrainedTokenizer): The tokenizer to use for generation.
            schema (Dict[str, Any]): The JSON schema to use for generation.
            prompt (str): The prompt to use for generation.
            debug (bool): Whether to print debug information.
            max_array_length (int): The maximum number of elements in an array.
            max_tokens_number (int): The maximum number of tokens in a number.
            max_tokens_string (int): The maximum number of tokens in a string.
            temperature (float): The temperature to use for generation.
            generation_marker (str): The marker used to track the current generation position in the JSON.
            num_retries (int): The maximum number of retries for generation.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.schema = schema
        self.prompt = prompt
        self.value = {}  # The JSON object being built
        self.current_field_context = None  # Track current field for comment injection

        # Configure generation parameters
        self.debug_on = debug
        self.max_array_length = max_array_length
        self.max_tokens_number = max_tokens_number
        self.max_tokens_string = max_tokens_string
        self.temperature = temperature
        self.num_retries = num_retries

        # Initialize token processors
        self.number_logit_processor = OutputNumbersTokens(
            tokenizer=self.tokenizer, prompt=self.prompt
        )
        self.array_end_logit_processor = OutputCommaAndBracketTokens(
            tokenizer=self.tokenizer, prompt=self.prompt
        )

        # Marker used to track where generation should happen
        self.generation_marker = generation_marker

    def debug(self, caller: str, value: str, is_prompt: bool = False) -> None:
        """
        Print debug information if debug mode is enabled.

        Args:
            caller (str): The name of the calling function
            value (str): The value to print
            is_prompt (bool): Whether the value is a prompt (affects coloring)
        """
        if not self.debug_on:
            return

        # Always print caller in green
        cprint(text=caller, color="green", end=" ")

        # Print value in yellow for prompts, blue otherwise
        color = "yellow" if is_prompt else "blue"
        cprint(text=value, color=color)

    def _build_field_guidance(self, schema: Dict[str, Any], field_name: str) -> str:
        """
        Build a guidance string for a specific field based on its schema.

        Args:
            schema (Dict[str, Any]): The schema for the field.
            field_name (str): The name of the field.

        Returns:
            str: A guidance string for the field.
        """
        guidance_parts = []

        # 1. Use explicit description first
        if "description" in schema:
            guidance_parts.append(schema["description"])

        # 2. Add constraint guidance
        constraints = []

        # Number constraints
        if schema.get("type") == "number":
            if "minimum" in schema and "maximum" in schema:
                constraints.append(
                    f"Must be between {schema['minimum']} and {schema['maximum']}"
                )
            elif "minimum" in schema:
                constraints.append(f"At least {schema['minimum']}")
            elif "maximum" in schema:
                constraints.append(f"At most {schema['maximum']}")

        # String constraints
        if schema.get("type") == "string":
            if "minLength" in schema and "maxLength" in schema:
                constraints.append(
                    f"Must be between {schema['minLength']} and {schema['maxLength']} characters"
                )
            elif "minLength" in schema:
                constraints.append(f"At least {schema['minLength']} characters")
            elif "maxLength" in schema:
                constraints.append(f"At most {schema['maxLength']} characters")

        # Combine constraints
        if constraints:
            guidance_parts.append(" | ".join(constraints))

        return " - ".join(guidance_parts) if guidance_parts else ""

    def _inject_comment_at_generation_point(
        self, json_progress: str, comment: str
    ) -> str:
        """
        Inject a comment at the generation point in the JSON progress.

        Args:
            json_progress (str): The JSON progress string.
            comment (str): The comment to inject.

        Returns:
            str: The JSON progress string with the comment injected.
        """

        if not comment:
            return json_progress

        # Find the generation marker
        marker_index = json_progress.find(f'"{self.generation_marker}"')

        if marker_index == -1:
            raise ValueError(
                f"Generation marker '{self.generation_marker}' not found in current progress"
            )

        # Look backwards to find where to inject the comment
        # We want want to place it right after the field name and colon

        # Find the last occurrence of ":" before the marker
        prefix = json_progress[:marker_index]
        colon_index = prefix.rfind(":")

        if colon_index == -1:
            # Fallback: inject right before the marker
            injection_point = marker_index
        else:
            # Look backwards from colon to find the field name
            # Skip whitespace before colon
            pos = colon_index - 1
            while pos >= 0 and json_progress[pos].isspace():
                pos -= 1

            # Should be at closing quote of field name
            if pos >= 0 and json_progress[pos] == '"':
                # Find the opening quote of field name
                opening_quote_index = json_progress.rfind('"', 0, pos)
                if opening_quote_index != -1:
                    injection_point = opening_quote_index
                else:
                    injection_point = marker_index
            else:
                injection_point = marker_index

        # Build the comment
        formatted_comment = f" /* {comment} */ "

        # Inject the comment
        return (
            json_progress[:injection_point]
            + formatted_comment
            + json_progress[injection_point:]
        )

    def get_prompt(self) -> str:
        """
        Get the current prompt with the in-progress JSON.

        This method constructs a prompt by combining:
        1. The original user prompt
        2. The JSON schema specification
        3. The current progress of JSON generation

        Returns:
            str: A formatted prompt string with the current JSON progress

        Raises:
            ValueError: If the generation marker is not found in the current progress
        """
        # Define template parts separately for clarity
        prompt_template = (
            "{prompt}\n"
            "Output result in the following JSON schema format:\n"
            "{schema}\n"
            "Result: {progress}"
        )

        # Build JSON progress
        json_progress = json.dumps(self.value)
        json_schema = json.dumps(self.schema)

        # Find marker position
        marker_index = json_progress.find(f'"{self.generation_marker}"')
        if marker_index == -1:
            raise ValueError(
                f"Generation marker '{self.generation_marker}' not found in current progress"
            )

        # Inject comment if we have current field context
        if self.current_field_context:
            field_name, field_schema = self.current_field_context
            guidance = self._build_field_guidance(
                schema=field_schema, field_name=field_name
            )
            if guidance:
                json_progress = self._inject_comment_at_generation_point(
                    json_progress=json_progress, comment=guidance
                )

                # Recalculate marker index after injection
                marker_index = json_progress.find(f'"{self.generation_marker}"')

        # Truncate progress at marker
        truncated_progress = json_progress[:marker_index]

        # Construct final prompt
        return prompt_template.format(
            prompt=self.prompt, schema=json_schema, progress=truncated_progress
        )

    def _is_valid_number(
        self,
        number: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> bool:
        """
        Check if a number satisfies the minimum and maximum constraints.

        Args:
            number (float): The number to validate
            min_val (Optional[float]): Minimum allowed value
            max_val (Optional[float]): Maximum allowed value

        Returns:
            bool: True if number is within constraints, False otherwise
        """
        if min_val is not None and number < min_val:
            return False
        if max_val is not None and number > max_val:
            return False
        return True

    def generate_number(
        self,
        schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        # max_retries: int = 3,
        temperature_multiplier: float = 1.3,
    ) -> float:
        """
        Generate a number value using the language model.

        Args:
            schema (Optional[Dict[str, Any]]): The schema with potential min/max constraints
            temperature (Optional[float]): Optional temperature override for generation
            # max_retries (int): Maximum number of retry attempts if generation fails
            temperature_multiplier (float): Factor to increase temperature by on retries

        Returns:
            float: The generated number value

        Raises:
            ValueError: If unable to generate a valid number after max retries
        """

        # Extract constraints
        min_val = schema.get("minimum") if schema else None
        max_val = schema.get("maximum") if schema else None

        if min_val or max_val:
            self.debug(
                caller="[generate_number]",
                value=f"Constraints: min={min_val}, max={max_val}, retries={self.num_retries}",
            )

        def _attempt_generation(
            current_temperature: float, attempt: int
        ) -> Optional[float]:
            # Get and debug the prompt
            prompt = self.get_prompt()
            self.debug(
                caller="[generate_number]",
                value=f"Attempt {attempt}: {prompt}",
                is_prompt=True,
            )

            # Prepare input tokens
            input_tokens = self.tokenizer.encode(text=prompt, return_tensors="pt").to(
                self.model.device
            )
            attention_mask = torch.ones_like(input_tokens)

            # Set base generation parameters
            generation_kwargs = {
                "inputs": input_tokens,
                "attention_mask": attention_mask,
                "max_new_tokens": self.max_tokens_number,
                "num_return_sequences": 1,
                "logits_processor": [self.number_logit_processor],
                "stopping_criteria": [
                    NumberStoppingCriteria(
                        tokenizer=self.tokenizer, prompt_length=len(input_tokens[0])
                    )
                ],
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            # Add sampling parameters only when temperature > 0
            if current_temperature > 0:
                generation_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": current_temperature,
                    }
                )
            else:
                generation_kwargs["do_sample"] = False
                generation_kwargs["temperature"] = None
                generation_kwargs["top_p"] = None
                generation_kwargs["top_k"] = None

            # Generate with constraints
            response = self.model.generate(**generation_kwargs)

            # Process response
            response_text = self.tokenizer.decode(response[0], skip_special_tokens=True)
            generated_part = response_text[len(prompt) :].strip().rstrip(".")
            self.debug(caller="[generate_number]", value=generated_part)

            try:
                return float(generated_part)
            except ValueError:
                return None

        # Initial attempt with base temperature
        current_temp = temperature or self.temperature
        attempt = 1

        while attempt <= self.num_retries + 1:
            number = _attempt_generation(
                current_temperature=current_temp, attempt=attempt
            )
            if number is not None:
                # Check constraints
                if self._is_valid_number(number, min_val, max_val):
                    self.debug(
                        caller="[generate_number]",
                        value=f"Success on attempt {attempt}: {number}",
                    )
                    return number
                else:
                    self.debug(
                        caller="[generate_number]",
                        value=f"Attempt {attempt}: {number} outside range [{min_val}, {max_val}]",
                    )
            else:
                self.debug(
                    caller="[generate_number]",
                    value=f"Attempt {attempt}: Failed to parse number",
                )

            # If not valid, retry
            attempt += 1
            current_temp *= temperature_multiplier

        # If we get here, all attempts failed
        constraint_msg = ""
        if min_val is not None or max_val is not None:
            constraint_msg = f" within range [{min_val}, {max_val}]"

        raise ValueError(
            f"Failed to generate a valid number{constraint_msg} after {self.num_retries} attempts"
        )

    def generate_boolean(self) -> bool:
        """
        Generate a boolean value using the language model.

        The method uses temperature-controlled generation and softmax probabilities
        to determine whether the output should be True or False.

        Returns:
            bool: The generated boolean value
        """
        prompt = self.get_prompt()
        self.debug(caller="[generate_boolean]", value=prompt, is_prompt=True)

        # Prepare input
        input_tensor = self.tokenizer.encode(text=prompt, return_tensors="pt").to(
            self.model.device
        )
        attention_mask = torch.ones_like(input_tensor)

        # Get model output with temperature for controlled randomness
        with torch.no_grad():
            outputs = self.model(input_tensor, attention_mask=attention_mask)
            logits = outputs.logits[0, -1] / self.temperature
            probs = torch.nn.functional.softmax(logits, dim=0)

        # Get token IDs for true/false variations, taking first token if multi-token
        true_tokens = ["true", "True"]
        false_tokens = ["false", "False"]

        true_ids = []
        false_ids = []

        for t in true_tokens:
            tokens = self.tokenizer.encode(t, add_special_tokens=False)
            if tokens:  # Only add if we got valid tokens
                true_ids.append(tokens[0])  # Take first token

        for f in false_tokens:
            tokens = self.tokenizer.encode(f, add_special_tokens=False)
            if tokens:  # Only add if we got valid tokens
                false_ids.append(tokens[0])  # Take first token

        # Sum probabilities for all true/false variations
        true_prob = sum(probs[tid].item() for tid in true_ids)
        false_prob = sum(probs[fid].item() for fid in false_ids)

        result = true_prob > false_prob
        self.debug(caller="[generate_boolean]", value=result)

        return result

    def generate_string(self) -> str:
        """
        Generate a string value using the language model.

        The method:
        1. Adds an opening quote to the prompt
        2. Generates text until a closing quote or max tokens is reached
        3. Processes the response to extract just the string content

        Returns:
            str: The generated string value, stripped of quotes and whitespace
        """
        # Handle enum values
        if "enum" in self.current_field_context[1]:
            return self.generate_enum(enum_values=self.current_field_context[1]["enum"])

        # Prepare prompt with opening quote
        prompt = self.get_prompt() + '"'
        self.debug(caller="[generate_string]", value=prompt, is_prompt=True)

        # Encode and move to model device
        input_tokens = self.tokenizer.encode(
            text=prompt,
            return_tensors="pt",
        ).to(self.model.device)
        attention_mask = torch.ones_like(input_tokens)

        # Set base generation parameters
        generation_kwargs = {
            "inputs": input_tokens,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_tokens_string,
            "num_return_sequences": 1,
            "stopping_criteria": [
                StringStoppingCriteria(
                    tokenizer=self.tokenizer, prompt_length=len(input_tokens[0])
                )
            ],
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Add sampling parameters only when temperature > 0
        if self.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": self.temperature,
                }
            )
        else:
            generation_kwargs["do_sample"] = False
            generation_kwargs["temperature"] = None
            generation_kwargs["top_p"] = None
            generation_kwargs["top_k"] = None

        # Generate with stopping criteria for closing quote
        response = self.model.generate(**generation_kwargs)

        # Extract generated tokens (excluding prompt if present)
        generated_tokens = response[0]
        if len(generated_tokens) > len(input_tokens[0]) and torch.equal(
            generated_tokens[: len(input_tokens[0])], input_tokens[0]
        ):
            generated_tokens = generated_tokens[len(input_tokens[0]) :]

        # Decode and clean up response
        decoded_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        self.debug(caller="[generate_string]", value=f"|{decoded_text}|")

        # Extract the string content (up to the closing quote if present)
        if '"' not in decoded_text:
            return decoded_text

        return decoded_text.split('"')[0].strip()

    def generate_enum(self, enum_values: List[str]) -> str:
        """
        Generate an enum value by selecting the most probable option from the allowed values.

        Args:
            enum_values (List[str]): List of allowed enum values

        Returns:
            str: The selected enum value with highest probability
        """
        prompt = self.get_prompt()
        self.debug(caller="[generate_enum]", value=prompt, is_prompt=True)

        # Prepare input tokens
        input_tokens = self.tokenizer.encode(text=prompt, return_tensors="pt").to(
            self.model.device
        )
        attention_mask = torch.ones_like(input_tokens)

        # Get model output
        with torch.no_grad():
            outputs = self.model(input_tokens, attention_mask=attention_mask)
            logits = outputs.logits[0, -1] / self.temperature
            probs = torch.nn.functional.softmax(logits, dim=0)

        # Calculate probabilities for each enum value
        enum_probabilities = {}
        for value in enum_values:
            try:
                tokens = self.tokenizer.encode(value, add_special_tokens=False)
                if tokens:
                    first_token_id = tokens[0]
                    enum_probabilities[value] = probs[first_token_id].item()
                else:
                    enum_probabilities[value] = 0.0
            except Exception:
                enum_probabilities[value] = 0.0

        # Select the most probable enum value
        selected_enum = max(enum_probabilities, key=enum_probabilities.get)

        self.debug(
            caller="[generate_enum]", value=f"Probabilities: {enum_probabilities}"
        )
        self.debug(caller="[generate_enum]", value=f"Selected: {selected_enum}")

        return selected_enum

    def generate_array(
        self,
        item_schema: Dict[str, Any],
        array: List[Any],
        array_schema: Dict[str, Any],
    ) -> List[Any]:
        """
        Generate an array with elements conforming to the item schema.

        The method generates array elements following these constraints:
        - If minItems is specified, generates at least that many elements
        - If maxItems is specified, generates at most that many elements
        - Falls back to model decision only when constraints allow it

        Args:
            item_schema (Dict[str, Any]): The schema defining the type and constraints for array items
            array (List[Any]): The array to populate with generated elements
            array_schema (Dict[str, Any]): The schema defining the array constraints

        Returns:
            List[Any]: The populated array with generated elements

        Raises:
            ValueError: If the item schema is invalid or generation fails
        """
        if not isinstance(item_schema, dict) or "type" not in item_schema:
            raise ValueError(
                "Invalid item schema: must be a dictionary with 'type' key"
            )

        # Extract constraints
        min_items = array_schema.get("minItems", 0)
        max_items = array_schema.get("maxItems", self.max_array_length)

        self.debug(
            caller="[generate_array]",
            value=f"Constraints: minItems={min_items}, maxItems={max_items}",
        )

        try:
            # Phase 1: Generate required elements (minItems)
            for i in range(min_items):
                self.debug(
                    caller="[generate_array]",
                    value=f"Generating required element {i+1}/{min_items}",
                )
                # Set context for array element
                if item_schema.get("type") in ("number", "boolean", "string"):
                    self.current_field_context = ("array item", item_schema)

                # Generate an element and add it to the array
                element = self.generate_value(schema=item_schema, obj=array)
                if array and array[-1] == self.generation_marker:
                    array[-1] = element
                else:
                    array.append(element)

            # Phase 2: Generate optional elements (between minItems and maxItems)
            if min_items < max_items:
                self.debug(
                    caller="[generate_array]",
                    value=f"Generating optional elements (up to {max_items} total)",
                )

                # Generate at least one element for the array
                for i in range(min_items, max_items):
                    # Set context for array element
                    if item_schema.get("type") in ("number", "boolean", "string"):
                        self.current_field_context = ("array item", item_schema)

                    # Continue: generate the next element
                    element = self.generate_value(schema=item_schema, obj=array)
                    if array and array[-1] == self.generation_marker:
                        array[-1] = element
                    else:
                        array.append(element)

                    # Check if we should add another element
                    array.append(self.generation_marker)
                    item_prompt = self.get_prompt()
                    array.pop()  # Remove the marker

                    try:
                        # Use LogitProcessor to force choice between "," and "]"
                        input_tokens = self.tokenizer.encode(
                            text=item_prompt, return_tensors="pt"
                        ).to(self.model.device)

                        attention_mask = torch.ones_like(input_tokens)

                        # Set base generation parameters
                        generation_kwargs = {
                            "inputs": input_tokens,
                            "attention_mask": attention_mask,
                            "max_new_tokens": 1,
                            "num_return_sequences": 1,
                            "logits_processor": [self.array_end_logit_processor],
                            "pad_token_id": self.tokenizer.eos_token_id,
                        }

                        # Add sampling parameters only when temperature > 0
                        if self.temperature > 0:
                            generation_kwargs.update(
                                {
                                    "do_sample": True,
                                    "temperature": self.temperature,
                                }
                            )
                        else:
                            generation_kwargs["do_sample"] = False
                            generation_kwargs["temperature"] = None
                            generation_kwargs["top_p"] = None
                            generation_kwargs["top_k"] = None

                        # Generate exactly one token, constrained to only "," and "]"
                        response = self.model.generate(**generation_kwargs)

                        # Extract the generated token
                        last_token = self.tokenizer.decode(
                            response[0][-1], skip_special_tokens=True
                        )
                        self.debug(
                            caller="[generate_array]",
                            value=f"Model chose: '{last_token}'",
                        )

                        # Stop if model chose closing bracket
                        if "]" in last_token:
                            break

                    except Exception as e:
                        self.debug(
                            caller="[generate_array]",
                            value=f"Error during array continuation: {str(e)}",
                        )
                        break  # Stop on error

            self.debug(
                caller="[generate_array]", value=f"Final array length: {len(array)}"
            )
            return array

        except Exception as e:
            self.debug(
                caller="[generate_array]", value=f"Error generating array: {str(e)}"
            )
            raise ValueError(f"Failed to generate array: {str(e)}")

    def generate_value(
        self,
        schema: Dict[str, Any],
        obj: Union[Dict[str, Any], List[Any]],
        key: Optional[str] = None,
    ) -> Any:
        """
        Generate a value according to the schema type.

        Args:
            schema (Dict[str, Any]): The schema defining the value type and constraints
            obj (Union[Dict[str, Any], List[Any]]): The parent object/array where the value will be stored
            key (Optional[str]): The property name if parent is an object, None if parent is an array

        Returns:
            Any: The generated value based on the schema type:
            - For primitives: number, boolean, or string
            - For arrays: List of generated elements
            - For objects: Dict of generated properties

        Raises:
            ValueError: If schema is missing type or type is unsupported
            KeyError: If required schema properties are missing
        """
        if not schema or "type" not in schema:
            raise ValueError("Schema must contain a 'type' field")

        schema_type = schema["type"]

        # Set context for field-specific guidance
        if schema_type in ("number", "boolean", "string") and key:
            self.current_field_context = (key, schema)

        # Helper function to set generation marker
        def set_marker():
            if key is not None:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)

        try:
            # Handle primitive types
            if schema_type in ("number", "boolean", "string"):
                set_marker()
                if schema_type == "number":
                    return self.generate_number(schema=schema)
                elif schema_type == "boolean":
                    return self.generate_boolean()
                else:  # string
                    return self.generate_string()

            # Handle arrays
            elif schema_type == "array":
                if "items" not in schema:
                    raise KeyError("Array schema must contain 'items' field")
                new_array = []
                if key is not None:
                    obj[key] = new_array
                else:
                    obj.append(new_array)
                return self.generate_array(
                    item_schema=schema["items"], array=new_array, array_schema=schema
                )

            # Handle objects
            elif schema_type == "object":
                if "properties" not in schema:
                    raise KeyError("Object schema must contain 'properties' field")
                new_obj = {}
                if key is not None:
                    obj[key] = new_obj
                else:
                    obj.append(new_obj)
                return self.generate_object(
                    properties=schema["properties"], obj=new_obj
                )

            else:
                raise ValueError(f"Unsupported schema type: {schema_type}")
        finally:
            # Clear context after generation (resource optimization)
            self.current_field_context = None

    def generate_object(
        self, properties: Dict[str, Any], obj: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an object by generating values for each property according to the schema.

        Args:
            properties (Dict[str, Any]): The schema properties defining the structure and types
                of values to generate. Each property should have a schema definition.
            obj (Dict[str, Any]): The object to populate with generated values.
                This will be modified in-place.

        Returns:
            Dict[str, Any]: The populated object with generated values for all properties.

        Raises:
            ValueError: If a property schema is invalid or generation fails for a property.
        """
        if not properties:
            return obj

        try:
            for key, schema in properties.items():
                if not isinstance(schema, dict):
                    raise ValueError(f"Invalid schema for property '{key}': {schema}")

                self.debug(
                    caller="[generate_object]", value=f"Generating value for '{key}'"
                )
                obj[key] = self.generate_value(schema=schema, obj=obj, key=key)

        except Exception as e:
            raise ValueError(f"Failed to generate object: {str(e)}") from e

        return obj

    def __call__(self) -> Dict[str, Any]:
        """
        Generate a complete JSON object according to the schema.

        This method:
        1. Validates that the schema is a valid object schema
        2. Resets the internal state
        3. Generates a new object according to the schema properties

        Returns:
            Dict[str, Any]: The generated JSON object conforming to the schema

        Raises:
            ValueError: If the schema is invalid or not an object schema
        """
        # Validate schema
        if not isinstance(self.schema, dict) or "properties" not in self.schema:
            raise ValueError("Schema must be an object schema with 'properties' field")

        # Reset internal state
        self.value = {}

        # Generate new object
        return self.generate_object(
            properties=self.schema["properties"], obj=self.value
        )
