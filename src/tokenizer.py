"""
Tokenizes text strings into tensors.
We will use the BERT tokenizer for now (our task is quite similar to the one in the BERT paper).
"""

from transformers import AutoTokenizer
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)  # not sure if we shouldn't better use the cased version instead


def tokenize_text(text: str) -> torch.Tensor:
    """
    Tokenizes a text string into a tensor.
    The tokenizer will also add a CLS token at the end of the sequence.

    Note:
        Usually, the CLS token is added at the beginning of the sequence, but I think that's somehow unnatural
        maybe that's just because I am used to appending it because of my work on vision transformers.
    """
    tokenized = tokenizer(text, return_tensors="pt", padding=False, truncation=False)[
        "input_ids"
    ].squeeze()  # remove the batch dimension

    tokenized_with_cls = torch.cat(
        [tokenized[1:], torch.tensor([tokenizer.cls_token_id])]
    )

    return tokenized_with_cls


def detokenize_text(tokens: torch.Tensor) -> str:
    """
    Converts a tensor of token IDs back into a text string.
    Handles the special case where we placed the CLS token at the end.
    """
    # Remove the CLS token from the end
    tokens_without_cls = tokens[:-1]

    # Add back the missing first token that we removed during tokenization
    full_tokens = torch.cat(
        [torch.tensor([tokenizer.cls_token_id]), tokens_without_cls]
    )

    # Convert to string
    text = tokenizer.decode(full_tokens, skip_special_tokens=True)
    return text


if __name__ == "__main__":
    print(
        f"The CLS token is: {tokenizer.cls_token}, its ID is: {tokenizer.cls_token_id}"
    )

    text = "The quick brown fox jumps over the lazy dog."
    tokenized = tokenize_text(text)
    print(tokenized)
