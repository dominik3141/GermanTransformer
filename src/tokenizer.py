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


def tokenize_texts(texts: list[str]) -> torch.Tensor:
    """
    Batch tokenizes text strings into a padded tensor.
    Each sequence will have a CLS token appended at the end.
    """
    # Tokenize all texts at once - this is more efficient than doing it one by one
    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,  # Enable padding for batch processing
        truncation=False,
    )["input_ids"]  # Shape: [batch_size, max_seq_len]

    # Remove the batch-added CLS tokens (first position)
    tokenized = tokenized[:, 1:]

    # Add CLS tokens at the end of each sequence
    batch_cls = torch.full(
        (len(texts), 1), tokenizer.cls_token_id, device=tokenized.device
    )
    tokenized_with_cls = torch.cat([tokenized, batch_cls], dim=1)

    return tokenized_with_cls


def detokenize_texts(token_ids: torch.Tensor) -> list[str]:
    """
    Converts batched token IDs back into text strings.
    Removes the CLS token from the end of each sequence.
    """
    # Remove CLS tokens from the end
    tokens_without_cls = token_ids[:, :-1]

    # Convert to list of lists for the tokenizer
    token_lists = tokens_without_cls.tolist()

    # Decode all sequences at once
    texts = tokenizer.batch_decode(
        token_lists,
        skip_special_tokens=True,  # Remove any remaining special tokens
        clean_up_tokenization_spaces=True,  # Clean up spaces around punctuation
    )

    return texts


if __name__ == "__main__":
    print(
        f"The CLS token is: {tokenizer.cls_token}, its ID is: {tokenizer.cls_token_id}"
    )

    text = "The quick brown fox jumps over the lazy dog."
    tokenized = tokenize_texts([text])
    decoded = detokenize_texts(tokenized)
    print(f"Original: {text}")
    print(f"Decoded:  {decoded[0]}")
