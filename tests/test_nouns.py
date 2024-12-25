from src.nouns import load_nouns_from_csv, create_train_val_dataloaders
import random
from src.tokenizer import tokenizer
import configparser


def test_create_train_val_dataloaders():
    """Tests that the train and validation dataloaders are created correctly"""

    batch_size = random.choice([16, 32, 64])

    nouns = load_nouns_from_csv("data/nouns_clean.csv")
    train_loader, val_loader = create_train_val_dataloaders(
        nouns, batch_size=batch_size
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0

    print(
        f"Train loader length: {len(train_loader)}, total number of training samples: {len(train_loader) * batch_size}"
    )
    print(
        f"Validation loader length: {len(val_loader)}, total number of validation samples: {len(val_loader) * batch_size}"
    )


def test_load_nouns_respects_max_length():
    """Verify that loaded nouns don't exceed the maximum sequence length"""

    config = configparser.ConfigParser()
    config.read("default.conf")
    max_sequence_length = int(config["MODEL"]["max_sequence_length"])

    nouns = load_nouns_from_csv("data/nouns_clean.csv")

    for noun in nouns:
        token_count = len(tokenizer.encode(noun.word)) + 1  # +1 for CLS token
        assert (
            token_count <= max_sequence_length
        ), f"Noun '{noun.word}' exceeds max length"


def test_valid_article():
    """Check that all articles are either m, f, or n"""

    nouns = load_nouns_from_csv("data/nouns_clean.csv")

    for noun in nouns:
        assert noun.article in [
            "m",
            "f",
            "n",
        ], f"Invalid article: {noun.article} for word: {noun.word}"
