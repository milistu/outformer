from typing import List
import torch
from transformers import PreTrainedTokenizer, LogitsWarper, StoppingCriteria


class StringStoppingCriteria(StoppingCriteria):
    """
    Stops string generation when a closing quote is encountered.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, prompt_length: int) -> None:
        """
        Args:
            tokenizer: The tokenizer to use.
            prompt_length: The length of the prompt.
        """
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """
        Args:
            input_ids: The input ids.
            scores: The scores.
            kwargs: Additional keyword arguments.

        Returns:
            bool: True if the generation should stop, False otherwise.
        """
        if input_ids.shape[1] < self.prompt_length:
            return False

        last_token = self.tokenizer.decode(
            token_ids=input_ids[0][-1], skip_special_tokens=True
        )

        return '"' in last_token


class NumberStoppingCriteria(StoppingCriteria):
    """
    Stops number generation when a complete number has been generated.
    A number is considered complete when:

        1. It contains more than one decimal point (invalid, so stop)
        2. It has a decimal point and has exceeded the specified precision
        3. A non-digit character like space or newline is found after digits
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_length: int,
        precision: int = 3,
    ) -> None:
        """
        Args:
            tokenizer: The tokenizer to use.
            prompt_length: The length of the prompt.
            precision: The precision of the number.
        """
        self.tokenizer = tokenizer
        self.precision = precision
        self.prompt_length = prompt_length

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """
        Args:
            input_ids: The input ids.
            scores: The scores.
            kwargs: Additional keyword arguments.

        Returns:
            bool: True if the generation should stop, False otherwise.
        """
        # Decode only the part after the prompt
        decoded = self.tokenizer.decode(
            token_ids=input_ids[0][self.prompt_length :], skip_special_tokens=True
        )

        # 1. Stop if there is more than one decimal point
        if decoded.count(".") > 1:
            return True

        # 2. Stop if it has a decimal point and has exceeded the specified precision
        if (
            decoded.count(".") == 1
            and len(decoded.strip().split(".")[1]) > self.precision
        ):
            return True

        # 3. Stop if a non-digit character like space or newline is found after digits
        if (
            len(decoded) > 1
            and any(c.isdigit() for c in decoded)
            and decoded[-1] in [" ", "\n"]
        ):
            return True

        return False


class OutputNumbersTokens(LogitsWarper):
    """
    Restricts token generation to only those that can be part of a valid number.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, prompt: str) -> None:
        """
        Args:
            tokenizer: The tokenizer to use.
            prompt: The prompt to use.
        """
        self.tokenizer = tokenizer
        self.tokenized_prompt = tokenizer(text=prompt, return_tensors="pt")
        self.prompt = prompt
        self.allowed_mask = self._create_validation_mask()

    def _create_validation_mask(self) -> torch.Tensor:
        """Create a mask of allowed tokens - digits and decimal point"""
        vocab_size = len(self.tokenizer)
        mask = torch.zeros(vocab_size, dtype=torch.bool)

        for _, token_id in self.tokenizer.get_vocab().items():
            token_str = self.tokenizer.decode(token_ids=token_id).strip()

            # Allow empty tokens and tokens containing only digits and at most one decimal point
            if token_str == "" or (
                all(c.isdigit() or c == "." for c in token_str)
                and token_str.count(".") <= 1
            ):
                mask[token_id] = True

        return mask

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids: The input ids.
            scores: The scores.

        Returns:
            torch.FloatTensor: The scores.
        """
        # Apply the mask to set scores of disallowed tokens to - inf
        mask = self.allowed_mask.expand_as(scores)
        scores[~mask] = -float("inf")

        return scores
