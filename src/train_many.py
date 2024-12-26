"""
We train mutliple models of different sizes and see how they perform.

We want to test the following models:
Name, Layers, Heads, d_model
tiny-a, 2, 2, 512
tiny-b, 2, 2, 1024
tiny-c, 2, 2, 2048
small-a, 4, 4, 512
small-b, 4, 4, 1024
small-c, 4, 4, 2048
medium-a, 8, 8, 512
medium-b, 8, 8, 1024
medium-c, 8, 8, 2048
large-a, 16, 16, 512
large-b, 16, 16, 1024
large-c, 16, 16, 2048

Batch size will be decided automatically to choose the largest batch size that fits in memory.
Learning rate will be dependent on the batch size. So far a learning rate of 0.000005 works well for a batch size of 64.
So we will use 0.000005 / 64 * batch_size as the learning rate for now.
"""

from dataclasses import dataclass
from typing import List
import torch
from src.train import train
import configparser


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    num_heads: int
    d_model: int


def get_max_batch_size(model_config: ModelConfig) -> int:
    """Finds the largest batch size that fits in memory for a given model configuration"""
    print(f"\nDetermining optimal batch size for model {model_config.name}...")
    test_batch_size = 1024

    while test_batch_size > 1:
        print(f"Trying batch size: {test_batch_size}")
        try:
            # Try training for just one batch with the current batch size
            train(
                num_layers=model_config.num_layers,
                num_heads=model_config.num_heads,
                d_model=model_config.d_model,
                batch_size=test_batch_size,
                test=True,
                max_batches=2,
                learning_rate=0.000005,  # doesn't matter here
                max_epochs=1,  # doesn't matter here
                dropout_rate=0.1,
                patience=3,
                min_delta=0.1,
                nouns_path="data/nouns_clean.csv",
            )
            print(f"Found optimal batch size: {test_batch_size}")
            return test_batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                test_batch_size //= 2
                torch.cuda.empty_cache()
                print("Out of memory, reducing batch size...")
            else:
                raise e
    return 1


def train_models() -> None:
    """Trains all model configurations with automatically determined batch sizes"""
    configs: List[ModelConfig] = [
        ModelConfig("tiny-a", 2, 2, 512),
        ModelConfig("tiny-b", 2, 2, 1024),
        ModelConfig("tiny-c", 2, 2, 2048),
        ModelConfig("small-a", 4, 4, 512),
        ModelConfig("small-b", 4, 4, 1024),
        ModelConfig("small-c", 4, 4, 2048),
        ModelConfig("medium-a", 8, 8, 512),
        ModelConfig("medium-b", 8, 8, 1024),
        ModelConfig("medium-c", 8, 8, 2048),
        ModelConfig("large-a", 16, 16, 512),
        ModelConfig("large-b", 16, 16, 1024),
        ModelConfig("large-c", 16, 16, 2048),
    ]

    config = configparser.ConfigParser()
    config.read("default.conf")

    base_lr = float(config["TRAINING"]["base_learning_rate"])
    max_epochs = int(config["TRAINING"]["max_epochs"])
    nouns_path = config["DATA"]["nouns_path"]
    dropout_rate = float(config["TRAINING"]["dropout_rate"])
    patience = int(config["TRAINING"]["patience"])
    min_delta = float(config["TRAINING"]["min_delta"])

    failed_models: List[tuple[str, str]] = []

    for config in configs:
        print(f"\n{'='*50}")
        print(f"Training model: {config.name}")
        print(f"{'='*50}")

        try:
            batch_size = get_max_batch_size(config)
            learning_rate = base_lr * batch_size

            train(
                model_name=config.name,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                d_model=config.d_model,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_epochs=max_epochs,
                nouns_path=nouns_path,
                dropout_rate=dropout_rate,
                patience=patience,
                min_delta=min_delta,
            )
        except Exception as e:
            error_msg = str(e)
            print(f"\nFAILED to train model {config.name}:")
            print(f"Error: {error_msg}")
            failed_models.append((config.name, error_msg))
            continue

    if failed_models:
        print("\n\nSummary of failed models:")
        print("========================")
        for model_name, error in failed_models:
            print(f"{model_name}: {error}")
    else:
        print("\n\nAll models trained successfully!")


if __name__ == "__main__":
    train_models()
