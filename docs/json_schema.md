# JSON Schema Parameters

Outformer currently supports a subset of JSON Schema parameters to define the structure and validation rules for generated outputs, with plans to expand support for additional parameters in future releases.

## Supported Parameters

### Type Definitions

- `type`: Specifies the data type of the value
  - Supported types: `string`, `number`, `integer`, `boolean`, `object`, `array`
  - Example:
    ```json
    {
        "type": "string",
        "description": "A text field"
    }
    ```

### Text Validation - Guidance

- `description`: Provides a description of the field (used for generation)
  - Example:
    ```json
    {
        "type": "string",
        "description": "The name of the person"
    }
    ```

### Array Validation

- `items`: Defines the schema for array elements
  - Example:
    ```json
    {
        "type": "array",
        "items": {
            "type": "string",
            "description": "Name of a participant"
        }
    }
    ```
- `minItems`: Minimum number of items required in the array
  - Example:
    ```json
    {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "string"
        }
    }
    ```
- `maxItems`: Maximum number of items allowed in the array
  - Example:
    ```json
    {
        "type": "array",
        "maxItems": 5,
        "items": {
            "type": "string"
        }
    }
    ```

### Object Properties

- `properties`: Defines the properties of an object
  - Example:
    ```json
    {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Person's name"
            },
            "age": {
                "type": "integer",
                "description": "Person's age"
            }
        }
    }
    ```

### Enums

- `enum`: Restricts values to a specific set of options
  - Example:
    ```json
    {
        "type": "string",
        "enum": ["option1", "option2", "option3"],
        "description": "Select one of the available options"
    }
    ```

## Usage Examples

### Basic Object Schema
```json
{
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "The name of the person"
        },
        "age": {
            "type": "integer",
            "description": "The age of the person"
        },
        "hobbies": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "string",
                "description": "A hobby or interest"
            }
        }
    }
}
```

### Complex Nested Schema
```json
{
    "type": "object",
    "properties": {
        "event": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the event"
                },
                "date": {
                    "type": "string",
                    "description": "Date of the event"
                },
                "participants": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Participant's name"
                            },
                            "role": {
                                "type": "string",
                                "enum": ["organizer", "attendee", "speaker"],
                                "description": "Role in the event"
                            }
                        }
                    }
                }
            }
        }
    }
}
```

## Best Practices

1. Always provide clear `description` fields to guide the model's generation, but note that longer descriptions may increase generation time
2. Use `minItems` and/or `maxItems` for arrays when you know exactly or approximately how many elements you need
3. Use `enum` to restrict values to specific predefined options
4. Maintain clean and well-organized schemas - this helps both human readability and LLM performance
5. Represent complex data structures using nested objects and arrays
6. If results are unsatisfactory, experiment with your prompt or consider using a larger model