"""
A minimal character-level tokenizer.
"""

import torch
from typing import Dict, List


class CharVocab:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.cls_token = "<cls>"

        # Create vocabulary (ASCII printable chars + special tokens)
        chars = [chr(i) for i in range(32, 127)]  # printable ASCII
        special_tokens = [self.pad_token, self.cls_token]
        self.char_to_id: Dict[str, int] = {
            char: idx for idx, char in enumerate(special_tokens + chars)
        }
        self.id_to_char: Dict[int, str] = {
            idx: char for char, idx in self.char_to_id.items()
        }

        self.cls_token_id = self.char_to_id[self.cls_token]
        self.pad_token_id = self.char_to_id[self.pad_token]
        self.vocab_size = len(self.char_to_id)


# Global tokenizer instance
tokenizer = CharVocab()


def tokenize_texts(texts: List[str]) -> torch.Tensor:
    """Batch tokenizes text strings into a padded tensor with CLS token at the end."""
    # Convert each text to character indices
    sequences = [
        [tokenizer.char_to_id.get(c, tokenizer.pad_token_id) for c in text]
        for text in texts
    ]

    # Find max length and pad sequences
    max_len = max(len(seq) for seq in sequences)
    padded = [
        seq + [tokenizer.pad_token_id] * (max_len - len(seq)) + [tokenizer.cls_token_id]
        for seq in sequences
    ]

    return torch.tensor(padded)


def detokenize_texts(token_ids: torch.Tensor) -> List[str]:
    """Converts batched token IDs back into text strings."""
    texts = []
    for seq in token_ids.tolist():
        # Remove CLS token and any padding
        valid_ids = [id for id in seq[:-1] if id != tokenizer.pad_token_id]
        text = "".join(tokenizer.id_to_char[id] for id in valid_ids)
        texts.append(text)
    return texts


if __name__ == "__main__":
    text = "Hello, World!"
    tokenized = tokenize_texts([text])
    decoded = detokenize_texts(tokenized)
    print(f"Original: {text}")
    print(f"Tokenized: {tokenized}")
    print(f"Decoded:  {decoded[0]}")
