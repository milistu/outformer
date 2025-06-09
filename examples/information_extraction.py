from transformers import AutoModelForCausalLM, AutoTokenizer

from outformer import Jsonformer, highlight_values


def main():
    model_name = "Qwen/Qwen3-1.7B"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    prompt = """
    Extract the event information.
    Alice and Bob are going to a science fair on Friday.
    """

    former = Jsonformer(model, tokenizer)

    event = former.generate(schema, prompt)

    highlight_values(event)

    # Expected Output:
    # {
    #     name: "Science Fair",
    #     date: "Friday",
    #     participants: ["Alice", "Bob"]
    # }


if __name__ == "__main__":
    main()
