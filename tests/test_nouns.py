from src.nouns import load_nouns_from_csv, create_train_val_dataloaders
import random


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
