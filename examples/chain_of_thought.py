from transformers import AutoModelForCausalLM, AutoTokenizer

from outformer import Jsonformer, highlight_values


def main():
    model_name = "Qwen/Qwen3-1.7B"
    cache_dir = ".cache"

    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    prompt = """
    You are a helpful math tutor. Guide the user through the solution step by step.
    how can I solve 8x + 7 = -23
    """

    former = Jsonformer(model, tokenizer, max_tokens_string=100)

    math_reasoning = former.generate(schema, prompt, debug=True)

    highlight_values(math_reasoning)

    # Expected Output:
    # {
    #     steps: [
    #         {
    #             explanation: "Subtract 7 from both sides to isolate the term with x.",
    #             output: "8x = -30",
    #         },
    #         {
    #             explanation: "Divide both sides by 8 to solve for x.",
    #             output: "x = -3.75",
    #         },
    #     ],
    #     final_answer: "x = -3.75",
    # }


if __name__ == "__main__":
    main()
