import random
import string
import torch

from src.tokenizer import tokenize_texts, tokenizer, detokenize_texts


def test_cls_append():
    """Make sure that the cls token is always appended"""
    text = "Hello, world!"
    tokenized = tokenize_texts([text])[0]  # Get first sequence
    assert tokenized[-1] == tokenizer.cls_token_id


def test_batch_cls_append():
    """Verify CLS token is appended to all sequences in a batch"""
    texts = ["Hello, world!", "Another text", "Third example"]
    tokenized = tokenize_texts(texts)
    assert all(seq[-1] == tokenizer.cls_token_id for seq in tokenized)


def test_tokenization():
    """Make sure that the detokenization is the inverse of the tokenization"""
    random_texts = [
        "".join(random.choices(string.ascii_letters, k=20)).lower() for _ in range(3)
    ]
    print(f"Random texts to test tokenization: {random_texts}")

    tokenized = tokenize_texts(random_texts)
    detokenized = detokenize_texts(tokenized)
    assert all(orig == decoded for orig, decoded in zip(random_texts, detokenized))


def test_batch_tokenization():
    """Verify batch tokenization produces correct shapes and padding"""
    texts = ["short", "a longer text", "the longest text in this batch"]
    tokenized = tokenize_texts(texts)

    # Check shape
    assert len(tokenized.shape) == 2
    assert tokenized.shape[0] == len(texts)

    # All sequences should have same length (padding)
    assert all(len(seq) == len(tokenized[0]) for seq in tokenized)


def test_single_vs_batch():
    """Ensure consistent tokenization regardless of batch size"""
    text = "test text"
    single = tokenize_texts([text])[0]
    batch = tokenize_texts([text, text])
    assert torch.equal(single, batch[0])
    assert torch.equal(batch[0], batch[1])


def test_empty_input():
    """Verify handling of empty strings"""
    texts = ["", "some text", ""]
    tokenized = tokenize_texts(texts)
    assert tokenized.shape[0] == len(texts)
    assert all(seq[-1] == tokenizer.cls_token_id for seq in tokenized)


def test_detokenize_empty():
    """Verify handling of empty sequences"""
    texts = ["", "some text", ""]
    tokenized = tokenize_texts(texts)
    detokenized = detokenize_texts(tokenized)
    assert len(detokenized) == len(texts)
    assert detokenized[1].strip() == "some text"  # Middle text should be preserved
