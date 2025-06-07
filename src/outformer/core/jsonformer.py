import json
from typing import Any, Dict, List, Optional, Union, Tuple

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
        *,
        debug: bool = False,
        max_array_length: int = 10,
        max_tokens_number: int = 6,
        max_tokens_string: int = 10,
        temperature: float = 0.7,
        generation_marker: str = "|GENERATION|",
        max_attempts: int = 3,
    ) -> None:
        """
        Initialize a Jsonformer instance.

        Args:
            model (PreTrainedModel): The model to use for generation
            tokenizer (PreTrainedTokenizer): The tokenizer to use for generation
            debug (bool): Whether to print debug information
            max_array_length (int): The maximum number of elements in an array
            max_tokens_number (int): The maximum number of tokens in a number
            max_tokens_string (int): The maximum number of tokens in a string
            temperature (float): The temperature to use for generation
            generation_marker (str): The marker used to track the current generation position in the JSON
            max_attempts (int): The maximum number of attempts for value generation (currently used in number generation)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.value = {}  # The JSON object being built
        self.current_schema = None

        self.prompt = None
        self.schema = None

        # Configure generation parameters
        self.debug_on = debug
        self.max_array_length = max_array_length
        self.max_tokens_number = max_tokens_number
        self.max_tokens_string = max_tokens_string
        self.temperature = temperature
        self.max_attempts = max_attempts

        # Marker used to track where generation should happen
        self.generation_marker = generation_marker

    def _debug(self, caller: str, value: str, is_prompt: bool = False) -> None:
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

    def _build_field_guidance(self, schema: Dict[str, Any]) -> str:
        """
        Build a guidance string for a specific field based on its schema.

        Args:
            schema (Dict[str, Any]): The schema for the field.

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

    def _get_prompt(self) -> str:
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
        if self.current_schema:
            guidance = self._build_field_guidance(schema=self.current_schema)
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

    def _get_generation_kwargs(
        self,
        input_tokens: torch.Tensor,
        max_new_tokens: int,
        logits_processor: Optional[List] = None,
        stopping_criteria: Optional[List] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get common generation parameters for model.generate() calls.

        Args:
            input_tokens (torch.Tensor): Input token tensor
            max_new_tokens (int): Maximum number of new tokens to generate
            logits_processor (Optional[List]): Optional list of logits processors
            stopping_criteria (Optional[List]): Optional list of stopping criteria
            temperature (Optional[float]): Optional temperature override for generation

        Returns:
            Dict[str, Any]: Generation parameters dictionary
        """
        temperature = temperature or self.temperature

        generation_kwargs = {
            "inputs": input_tokens,
            "attention_mask": torch.ones_like(input_tokens),
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if logits_processor:
            generation_kwargs["logits_processor"] = logits_processor
        if stopping_criteria:
            generation_kwargs["stopping_criteria"] = stopping_criteria

        # Add sampling parameters only when temperature > 0
        if temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                }
            )
        else:
            generation_kwargs.update(
                {
                    "do_sample": False,
                    "temperature": None,
                    "top_p": None,
                    "top_k": None,
                }
            )

        return generation_kwargs

    def _process_tokens(
        self,
        prompt: str,
        max_new_tokens: int,
        logits_processor: Optional[List] = None,
        stopping_criteria: Optional[List] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, str]:
        """
        Process tokens for generation, including encoding and generation.

        Args:
            prompt (str): The input prompt
            max_new_tokens (int): Maximum number of new tokens to generate
            logits_processor (Optional[List]): Optional list of logits processors
            stopping_criteria (Optional[List]): Optional list of stopping criteria
            temperature (Optional[float]): Optional temperature override for generation

        Returns:
            Tuple[torch.Tensor, str]: Generated tokens and decoded text
        """
        input_tokens = self.tokenizer.encode(text=prompt, return_tensors="pt").to(
            self.model.device
        )

        generation_kwargs = self._get_generation_kwargs(
            input_tokens=input_tokens,
            max_new_tokens=max_new_tokens,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            temperature=temperature,
        )

        response = self.model.generate(**generation_kwargs)

        # Extract generated tokens (excluding prompt if present)
        generated_tokens = response[0]
        if len(generated_tokens) > len(input_tokens[0]) and torch.equal(
            generated_tokens[: len(input_tokens[0])], input_tokens[0]
        ):
            generated_tokens = generated_tokens[len(input_tokens[0]) :]

        response_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return generated_tokens, response_text

    def _generate_number(
        self,
        schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        temperature_multiplier: float = 1.3,
    ) -> float:
        """
        Generate a number value using the language model.

        Args:
            schema (Optional[Dict[str, Any]]): The schema with potential min/max constraints
            temperature (Optional[float]): Optional temperature override for generation
            temperature_multiplier (float): Factor to increase temperature by on each attempt

        Returns:
            float: The generated number value

        Raises:
            ValueError: If unable to generate a valid number after max attempts
        """

        # Extract constraints
        min_val = schema.get("minimum") if schema else None
        max_val = schema.get("maximum") if schema else None

        if min_val or max_val:
            self._debug(
                caller="[generate_number]",
                value=f"Constraints: min={min_val}, max={max_val}, attempts={self.max_attempts}",
            )

        def _attempt_generation(
            current_temperature: float, attempt: int
        ) -> Optional[float]:
            prompt = self._get_prompt()
            self._debug(
                caller="[generate_number]",
                value=f"Attempt {attempt}: {prompt}",
                is_prompt=True,
            )

            _, response_text = self._process_tokens(
                prompt=prompt,
                max_new_tokens=self.max_tokens_number,
                logits_processor=[OutputNumbersTokens(tokenizer=self.tokenizer)],
                stopping_criteria=[
                    NumberStoppingCriteria(
                        tokenizer=self.tokenizer,
                        prompt_length=len(self.tokenizer.encode(prompt)),
                    )
                ],
                temperature=current_temperature,
            )

            generated_part = response_text.strip().rstrip(".")
            self._debug(caller="[generate_number]", value=generated_part)

            try:
                return float(generated_part)
            except ValueError:
                return None

        # Initial attempt with base temperature
        current_temp = temperature or self.temperature
        attempt = 1

        while attempt <= self.max_attempts + 1:
            number = _attempt_generation(
                current_temperature=current_temp, attempt=attempt
            )
            if number is not None:
                # Check constraints
                if self._is_valid_number(number, min_val, max_val):
                    self._debug(
                        caller="[generate_number]",
                        value=f"Success on attempt {attempt}: {number}",
                    )
                    return number
                else:
                    self._debug(
                        caller="[generate_number]",
                        value=f"Attempt {attempt}: {number} outside range [{min_val}, {max_val}]",
                    )
            else:
                self._debug(
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
            f"Failed to generate a valid number{constraint_msg} after {self.max_attempts} attempts"
        )

    def _generate_boolean(self) -> bool:
        """
        Generate a boolean value using the language model.

        The method uses temperature-controlled generation and softmax probabilities
        to determine whether the output should be True or False.

        Returns:
            bool: The generated boolean value
        """
        prompt = self._get_prompt()
        self._debug(caller="[generate_boolean]", value=prompt, is_prompt=True)

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
        self._debug(caller="[generate_boolean]", value=result)

        return result

    def _generate_string(self) -> str:
        """
        Generate a string value using the language model.

        The method:
        1. Adds an opening quote to the prompt
        2. Generates text until a closing quote or max tokens is reached
        3. Processes the response to extract just the string content

        Returns:
            str: The generated string value, stripped of quotes and whitespace
        """
        if self.current_schema and "enum" in self.current_schema:
            return self._generate_enum(enum_values=self.current_schema["enum"])

        # Prepare prompt with opening quote
        prompt = self._get_prompt() + '"'
        self._debug(caller="[generate_string]", value=prompt, is_prompt=True)

        _, response_text = self._process_tokens(
            prompt=prompt,
            max_new_tokens=self.max_tokens_string,
            stopping_criteria=[
                StringStoppingCriteria(
                    tokenizer=self.tokenizer,
                    prompt_length=len(self.tokenizer.encode(prompt)),
                )
            ],
        )

        self._debug(caller="[generate_string]", value=f"|{response_text}|")

        if '"' not in response_text:
            return response_text

        return response_text.split('"')[0].strip()

    def _generate_enum(self, enum_values: List[str]) -> str:
        """
        Generate an enum value token by token and selecting the most probable option from the allowed values.

        Args:
            enum_values (List[str]): List of allowed enum values

        Returns:
            str: The selected enum value with highest probability

        Raises:
            ValueError: If the enum values list is empty
        """
        if not enum_values:
            raise ValueError("Enum values list cannot be empty")

        # Get all possible tokens for each enum value
        enum_tokens = {}
        for value in enum_values:
            try:
                tokens = self.tokenizer.encode(value, add_special_tokens=False)
                if tokens:
                    enum_tokens[value] = tokens
                else:
                    enum_tokens[value] = []
            except Exception:
                enum_tokens[value] = []

        enum_tokens = {k: v for k, v in enum_tokens.items() if v}
        if not enum_tokens:
            raise ValueError("No valid enum values could be tokenized")

        # Keep track of possible matches as we generate tokens
        possible_matches = list(enum_tokens.keys())
        generated_tokens = []

        while possible_matches:
            prompt = self._get_prompt()
            if generated_tokens:
                # Add already generated tokens to the prompt
                prompt += self.tokenizer.decode(generated_tokens)

            self._debug(caller="[generate_enum]", value=prompt, is_prompt=True)

            input_tokens = self.tokenizer.encode(text=prompt, return_tensors="pt").to(
                self.model.device
            )
            attention_mask = torch.ones_like(input_tokens)

            with torch.no_grad():
                outputs = self.model(input_tokens, attention_mask=attention_mask)
                logits = outputs.logits[0, -1] / self.temperature
                probs = torch.nn.functional.softmax(logits, dim=0)

            # Get the next token for each possible match
            next_token_probs = {}
            for value in possible_matches:
                tokens = enum_tokens[value]
                if len(generated_tokens) < len(tokens):
                    next_token = tokens[len(generated_tokens)]
                    next_token_probs[value] = probs[next_token].item()
                else:
                    # This value has been fully generated
                    next_token_probs[value] = 1.0

            selected_value = max(next_token_probs, key=next_token_probs.get)
            selected_tokens = enum_tokens[selected_value]

            # If we've generated all tokens for the selected value, we're done
            if len(generated_tokens) >= len(selected_tokens):
                break

            # Add the next token to our generated sequence
            next_token = selected_tokens[len(generated_tokens)]
            generated_tokens.append(next_token)

            self._debug(
                caller="[generate_enum]",
                value=f"Generated tokens: '{self.tokenizer.decode(generated_tokens)}', "
                f"Possible matches: {possible_matches}",
            )

            # Update possible matches based on the generated token
            possible_matches = [
                value
                for value in possible_matches
                if len(enum_tokens[value]) > len(generated_tokens)
                and enum_tokens[value][len(generated_tokens) - 1] == next_token
            ]

        self._debug(caller="[generate_enum]", value=f"Selected: {selected_value}")
        return selected_value

    def _generate_array(
        self,
        schema: Dict[str, Any],
        array: List[Any],
    ) -> List[Any]:
        """
        Generate an array with elements conforming to the item schema.

        The method generates array elements following these constraints:
        - If minItems is specified, generates at least that many elements
        - If maxItems is specified, generates at most that many elements
        - Falls back to model decision only when constraints allow it

        Args:
            schema (Dict[str, Any]): The schema defining the array constraints
            array (List[Any]): The array to populate with generated elements

        Returns:
            List[Any]: The populated array with generated elements

        Raises:
            ValueError: If the item schema is invalid or generation fails
        """
        if "items" not in schema:
            raise ValueError("Array schema must contain 'items' field")

        if not isinstance(schema["items"], dict) or "type" not in schema["items"]:
            raise ValueError(
                "Invalid item schema: must be a dictionary with 'type' key"
            )

        # Extract constraints
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", self.max_array_length)

        self._debug(
            caller="[generate_array]",
            value=f"Constraints: minItems={min_items}, maxItems={max_items}",
        )

        try:
            # Phase 1: Generate required elements (minItems)
            for i in range(min_items):
                self._debug(
                    caller="[generate_array]",
                    value=f"Generating required element {i+1}/{min_items}",
                )

                # Generate an element and add it to the array
                element = self._generate_value(schema=schema["items"], obj=array)
                array[-1] = element

            # Phase 2: Generate optional elements (between minItems and maxItems)
            if min_items < max_items:
                self._debug(
                    caller="[generate_array]",
                    value=f"Generating optional elements (up to {max_items} total)",
                )

                # Generate optional elements (up to maxItems total)
                for i in range(min_items, max_items):
                    # After inserting the element, decide if we should keep going
                    array.append(self.generation_marker)
                    item_prompt = self._get_prompt()
                    array.pop()

                    try:
                        # Use LogitProcessor to force choice between "," and "]"
                        _, response_text = self._process_tokens(
                            prompt=item_prompt,
                            max_new_tokens=1,
                            logits_processor=[
                                OutputCommaAndBracketTokens(tokenizer=self.tokenizer)
                            ],
                        )

                        self._debug(
                            caller="[generate_array]",
                            value=f"Model chose: '{response_text}'",
                        )

                        if "]" in response_text:
                            break

                        # Continue: generate the next element
                        element = self._generate_value(
                            schema=schema["items"], obj=array
                        )
                        array[-1] = element

                    except Exception as e:
                        self._debug(
                            caller="[generate_array]",
                            value=f"Error during array continuation: {str(e)}",
                        )
                        break

            self._debug(
                caller="[generate_array]", value=f"Final array length: {len(array)}"
            )
            return array

        except Exception as e:
            self._debug(
                caller="[generate_array]", value=f"Error generating array: {str(e)}"
            )
            raise ValueError(f"Failed to generate array: {str(e)}")

    def _generate_value(
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
        if schema_type in ("number", "boolean", "string"):
            self.current_schema = schema

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
                    return self._generate_number(schema=schema)
                elif schema_type == "boolean":
                    return self._generate_boolean()
                else:  # string
                    return self._generate_string()

            # Handle arrays
            elif schema_type == "array":
                if "items" not in schema:
                    raise KeyError("Array schema must contain 'items' field")
                new_array = []
                if key is not None:
                    obj[key] = new_array
                else:
                    obj.append(new_array)
                return self._generate_array(schema=schema, array=new_array)

            # Handle objects
            elif schema_type == "object":
                if "properties" not in schema:
                    raise KeyError("Object schema must contain 'properties' field")
                new_obj = {}
                if key is not None:
                    obj[key] = new_obj
                else:
                    obj.append(new_obj)
                return self._generate_object(
                    properties=schema["properties"], obj=new_obj
                )

            else:
                raise ValueError(f"Unsupported schema type: {schema_type}")
        finally:
            # Clear context after generation (resource optimization)
            self.current_schema = None

    def _generate_object(
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

                self._debug(
                    caller="[generate_object]", value=f"Generating value for '{key}'"
                )
                obj[key] = self._generate_value(schema=schema, obj=obj, key=key)

        except Exception as e:
            raise ValueError(f"Failed to generate object: {str(e)}") from e

        return obj

    def _validate_schema(
        self, schema: Dict[str, Any], required_fields: List[str] = None
    ) -> None:
        """
        Validate a schema against required fields and type constraints.

        Args:
            schema (Dict[str, Any]): The schema to validate
            required_fields (List[str], optional): List of required fields in the schema

        Raises:
            ValueError: If schema is invalid or missing required fields
        """
        if not isinstance(schema, dict):
            raise ValueError("Schema must be a dictionary")

        if required_fields:
            missing_fields = [field for field in required_fields if field not in schema]
            if missing_fields:
                raise ValueError(
                    f"Schema missing required fields: {', '.join(missing_fields)}"
                )

        if "type" in schema and schema["type"] not in [
            "number",
            "boolean",
            "string",
            "array",
            "object",
        ]:
            raise ValueError(f"Invalid schema type: {schema['type']}")

        if schema.get("type") == "array" and "items" not in schema:
            raise ValueError("Array schema must contain 'items' field")

        if schema.get("type") == "object" and "properties" not in schema:
            raise ValueError("Object schema must contain 'properties' field")

    def generate(
        self,
        schema: Dict[str, Any],
        prompt: str,
        *,
        debug: Optional[bool] = None,
        max_array_length: Optional[int] = None,
        max_tokens_number: Optional[int] = None,
        max_tokens_string: Optional[int] = None,
        temperature: Optional[float] = None,
        max_attempts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a JSON object according to the schema and prompt.

        Args:
            schema (Dict[str, Any]): The schema defining the JSON structure
            prompt (str): The prompt guiding the generation
            debug (Optional[bool]): Whether to enable debug mode
            max_array_length (Optional[int]): The maximum length of arrays to generate
            max_tokens_number (Optional[int]): The maximum number of tokens to generate for numbers
            max_tokens_string (Optional[int]): The maximum number of tokens to generate for strings
            temperature (Optional[float]): The temperature for the generation
            max_attempts (Optional[int]): The maximum number of attempts for value generation (currently used in number generation)

        Returns:
            Dict[str, Any]: The generated JSON object conforming to the schema

        Raises:
            ValueError: If schema is invalid or prompt is empty
        """
        self._validate_schema(schema, required_fields=["properties"])
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        # Reset internal state
        self.value = {}

        # Update instance variables
        self.schema = schema
        self.prompt = prompt
        self.debug_on = debug if debug is not None else self.debug_on
        self.max_array_length = (
            max_array_length if max_array_length is not None else self.max_array_length
        )
        self.max_tokens_number = (
            max_tokens_number
            if max_tokens_number is not None
            else self.max_tokens_number
        )
        self.max_tokens_string = (
            max_tokens_string
            if max_tokens_string is not None
            else self.max_tokens_string
        )
        self.temperature = temperature if temperature is not None else self.temperature
        self.max_attempts = (
            max_attempts if max_attempts is not None else self.max_attempts
        )

        return self._generate_object(
            properties=self.schema["properties"], obj=self.value
        )
