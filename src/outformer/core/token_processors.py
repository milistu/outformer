import torch
from transformers import LogitsProcessor, PreTrainedTokenizer, StoppingCriteria


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


class OutputNumbersTokens(LogitsProcessor):
    """
    Restricts token generation to only those that can be part of a valid number.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        """
        Args:
            tokenizer: The tokenizer to use.
        """
        self.tokenizer = tokenizer
        self.allowed_tokens = self._get_allowed_tokens()

    def _get_allowed_tokens(self) -> set[int]:
        """Create a set of allowed token IDs - digits and decimal point"""
        allowed_tokens = set()

        # Add special tokens that might be needed
        special_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        ]
        allowed_tokens.update(t for t in special_tokens if t is not None)

        # Add tokens that represent digits and decimal point
        for token_id in range(self.tokenizer.vocab_size):
            try:
                token_str = self.tokenizer.decode(token_ids=token_id).strip()
                # Allow empty tokens and tokens containing only digits and at most one decimal point
                if token_str == "" or (
                    all(c.isdigit() or c == "." for c in token_str)
                    and token_str.count(".") <= 1
                ):
                    allowed_tokens.add(token_id)
            except Exception:
                continue  # Skip tokens that can't be decoded

        return allowed_tokens

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
            torch.FloatTensor: The scores with disallowed tokens masked.
        """
        # Create a mask for allowed tokens
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for token_id in self.allowed_tokens:
            if token_id < scores.shape[-1]:  # Ensure token_id is within vocabulary size
                mask[..., token_id] = True

        # Set scores of disallowed tokens to -inf
        scores[~mask] = -float("inf")

        return scores


class OutputCommaAndBracketTokens(LogitsProcessor):
    """
    LogitsProcessor that constrains generation to only comma and closing bracket tokens.

    This processor is specifically used in array generation to determine whether to:
    1. Continue the array (when comma is generated)
    2. End the array (when closing bracket is generated)

    It ensures that the model can only choose between these two structural elements,
    preventing any other tokens from being generated at array element boundaries.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        """
        Args:
            tokenizer: The tokenizer to use.
        """
        self.tokenizer = tokenizer
        self.allowed_tokens = self._get_allowed_tokens()

    def _get_allowed_tokens(self) -> set[int]:
        """Create a set of allowed token IDs - comma and closing bracket"""
        allowed_tokens = set()

        # Add special tokens that might be needed
        special_tokens = [
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        ]
        allowed_tokens.update(t for t in special_tokens if t is not None)

        # Find tokens that are exactly "," or "]"
        for token_id in range(self.tokenizer.vocab_size):
            try:
                token_str = self.tokenizer.decode(token_ids=token_id).strip()

                if token_str in [",", "]"]:
                    allowed_tokens.add(token_id)
            except Exception:
                continue  # Skip tokens that can't be decoded

        return allowed_tokens

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids: The input ids.
            scores: The scores.

        Returns:
            torch.FloatTensor: The scores with only comma and bracket tokens allowed.
        """
        # Create a mask for allowed tokens
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for token_id in self.allowed_tokens:
            if token_id < scores.shape[-1]:  # Ensure token_id is within vocabulary size
                mask[..., token_id] = True

        # Set scores of disallowed tokens to -inf
        scores[~mask] = -float("inf")

        return scores
