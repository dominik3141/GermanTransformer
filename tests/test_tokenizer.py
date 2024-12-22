import random
import string

from src.tokenizer import detokenize_text, tokenize_text, tokenizer


def test_cls_append():
    """Make sure that the cls token is always appended"""

    text = "Hello, world!"
    tokenized = tokenize_text(text)
    assert tokenized[-1] == tokenizer.cls_token_id


def test_tokenization():
    """Make sure that the detokenization is the inverse of the tokenization"""

    random_text = "".join(random.choices(string.ascii_letters, k=20)).lower()

    print(f"Random text to test tokenization: {random_text}")

    tokenized = tokenize_text(random_text)
    detokenized = detokenize_text(tokenized)
    assert random_text == detokenized
