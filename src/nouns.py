"""
Define a class for nouns and add dataset and dataloader.
"""

import csv
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple
import torch
import configparser
from src.tokenizer import tokenizer


class Noun:
    """
    A noun is a string together with an article
    """

    def __init__(self, word: str, article: str):
        self.word = word
        self.article = article  # "m", "f", "n"

    def __str__(self):
        return f"{self.word}, {self.article}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.word == other.word and self.article == other.article


def load_nouns_from_csv(path: str) -> list[Noun]:
    """
    Load nouns from a csv file, skipping those that would exceed the maximum sequence length
    when tokenized. Returns only the nouns that fit within the length constraint.
    """
    # Read max sequence length from config
    config = configparser.ConfigParser()
    config.read("default.conf")
    max_sequence_length = int(config["MODEL"]["max_sequence_length"])

    valid_nouns: list[Noun] = []

    with open(path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            # Get token count for the word
            token_count = len(tokenizer.encode(row[0])) + 1  # +1 for CLS token

            if token_count <= max_sequence_length:
                valid_nouns.append(Noun(*row))

    return valid_nouns


class NounDataset(Dataset):
    """Dataset of German nouns and their articles"""

    def __init__(self, nouns: list[Noun], transform: Optional[callable] = None):
        self.nouns = nouns
        self.transform = transform

        # Create article to index mapping
        self.article_to_idx = {"m": 0, "f": 1, "n": 2}

    def __len__(self) -> int:
        return len(self.nouns)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        noun = self.nouns[idx]
        word = noun.word
        article_idx = self.article_to_idx[noun.article]

        if self.transform:
            word = self.transform(word)

        return word, article_idx


def create_train_val_dataloaders(
    nouns: list[Noun],
    val_ratio: float = 0.2,
    batch_size: int = 32,
    transform: Optional[callable] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Creates separate train and validation DataLoaders with the specified split ratio"""
    dataset = NounDataset(nouns, transform)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    # Create a generator with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
