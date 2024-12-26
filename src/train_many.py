"""
We train mutliple models of different sizes and see how they perform.

We want to test the following models:
Name, Layers, Heads, d_model
tiny-a, 2, 2, 256
tiny-b, 2, 2, 512
tiny-c, 2, 2, 1024
tiny-d, 2, 2, 2048
small-a, 4, 4, 256
small-b, 4, 4, 512
small-c, 4, 4, 1024
small-d, 4, 4, 2048
medium-a, 8, 8, 256
medium-b, 8, 8, 512
medium-c, 8, 8, 1024
medium-d, 8, 8, 2048
large-a, 16, 16, 256
large-b, 16, 16, 512
large-c, 16, 16, 1024
large-d, 16, 16, 2048

Batch size will be decided automatically to choose the largest batch size that fits in memory.
Learning rate will be dependent on the batch size. So far a learning rate of 0.000005 works well for a batch size of 64.
So we will use 0.000005 / 64 * batch_size as the learning rate for now.
"""

from dataclasses import dataclass
from typing import List
import torch
from src.train import train
from src.nn import TransformerEncoder
import configparser
import wandb
import math
import traceback


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    num_heads: int
    d_model: int


def get_model_stats(model_config: ModelConfig) -> tuple[int, int]:
    """Finds the largest batch size that fits in memory and counts parameters"""
    print(f"\nDetermining optimal batch size for model {model_config.name}...")
    test_batch_size = 1024
    MAX_MEMORY_USAGE = 0.8  # Use only 80% of available GPU memory
    n_params = 0

    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory

    while test_batch_size > 1:
        print(f"Trying batch size: {test_batch_size}")
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            train(
                num_layers=model_config.num_layers,
                num_heads=model_config.num_heads,
                d_model=model_config.d_model,
                batch_size=test_batch_size,
                test=True,
                max_batches=10,
                learning_rate=0.000005,
                max_epochs=1,
                dropout_rate=0.1,
                patience=3,
                min_delta=0.1,
                nouns_path="data/nouns_clean.csv",
            )

            # instantiate the model to count parameters
            model = TransformerEncoder(
                num_layers=model_config.num_layers,
                num_heads=model_config.num_heads,
                d_model=model_config.d_model,
                dropout_rate=0.1,
                max_sequence_length=512,
                num_classes=3,
            )

            # Count parameters
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Check memory usage
            memory_used = torch.cuda.max_memory_allocated()
            memory_usage_ratio = memory_used / total_memory

            if memory_usage_ratio > MAX_MEMORY_USAGE:
                print(
                    f"Memory usage too high ({memory_usage_ratio:.1%}), reducing batch size..."
                )
                test_batch_size //= 2
                continue

            print(
                f"Found optimal batch size: {test_batch_size} (memory usage: {memory_usage_ratio:.1%})"
            )
            return test_batch_size, n_params

        except RuntimeError as e:
            if "out of memory" in str(e):
                test_batch_size //= 2
                torch.cuda.empty_cache()
                print("Out of memory, reducing batch size...")
            else:
                raise e
    return 1, n_params


def get_learning_rate(
    config: ModelConfig, batch_size: int, n_params: int, base_lr: float = 5e-6
) -> float:
    """
    Calculates learning rate based on both batch size and parameter count.
    Uses linear scaling for batch size and square root scaling for parameters.
    """
    base_batch_size = 64
    base_params = 5_000_000  # 5M parameters as reference

    # Linear scaling with batch size, square root scaling with parameters
    batch_scale = batch_size / base_batch_size
    param_scale = math.sqrt(base_params / n_params)

    return base_lr * batch_scale * param_scale


def train_models() -> None:
    """Trains all model configurations with automatically determined batch sizes"""
    configs: List[ModelConfig] = [
        ModelConfig("tiny-a", 2, 2, 512),
        ModelConfig("tiny-b", 2, 2, 1024),
        ModelConfig("tiny-c", 2, 2, 2048),
        ModelConfig("small-a", 4, 4, 512),
        ModelConfig("small-b", 4, 4, 1024),
        ModelConfig("small-c", 4, 4, 2048),
        ModelConfig("special-a", 32, 4, 32),
        ModelConfig("special-b", 32, 4, 64),
        ModelConfig("special-c", 32, 4, 128),
        ModelConfig("special-d", 32, 4, 256),
        ModelConfig("special-e", 32, 4, 512),
        ModelConfig("special-f", 32, 4, 1024),
        ModelConfig("medium-a", 8, 4, 512),
        ModelConfig("medium-b", 8, 4, 1024),
        ModelConfig("medium-c", 8, 4, 2048),
        ModelConfig("large-a", 16, 4, 512),
        ModelConfig("large-b", 16, 4, 1024),
        ModelConfig("large-c", 16, 4, 2048),
    ]

    config = configparser.ConfigParser()
    config.read("default.conf")

    base_lr = float(config["TRAINING"]["base_learning_rate"])
    max_epochs = int(config["TRAINING"]["max_epochs"])
    nouns_path = config["DATA"]["nouns_path"]
    dropout_rate = float(config["TRAINING"]["dropout_rate"])
    patience = int(config["TRAINING"]["patience"])
    min_delta = float(config["TRAINING"]["min_delta"])
    val_ratio = float(config["TRAINING"]["val_ratio"])

    failed_models: List[tuple[str, str]] = []

    for config in configs:
        print(f"\n{'='*50}")
        print(f"Training model: {config.name}")
        print(f"{'='*50}")

        batch_size, n_params = get_model_stats(config)
        learning_rate = get_learning_rate(config, batch_size, n_params)

        print(f"Model parameters: {n_params:,}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate:.2e}")

        while batch_size >= 1:
            try:
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
                    val_ratio=val_ratio,
                )
                # If training succeeds, break the retry loop
                break

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(
                        f"\nOut of memory with batch size {batch_size}, retrying with smaller batch..."
                    )
                    # Clean up wandb run if it exists
                    if wandb.run is not None:
                        wandb.finish(exit_code=1)
                    # Reduce batch size and learning rate proportionally
                    batch_size //= 2
                    learning_rate = get_learning_rate(config, batch_size, n_params)
                    torch.cuda.empty_cache()
                    continue
                # For non-memory RuntimeErrors, show traceback and record failure
                error_msg = (
                    f"\nFAILED to train model {config.name}:\n{traceback.format_exc()}"
                )
                print(error_msg)
                failed_models.append((config.name, str(e)))
                if wandb.run is not None:
                    wandb.finish(exit_code=1)
                break
            except Exception as e:
                # For all other exceptions, show traceback and record failure
                error_msg = (
                    f"\nFAILED to train model {config.name}:\n{traceback.format_exc()}"
                )
                print(error_msg)
                failed_models.append((config.name, str(e)))
                if wandb.run is not None:
                    wandb.finish(exit_code=1)
                break

        # If we exhausted all batch sizes, add to failed models
        if batch_size < 1:
            error_msg = "Failed to find working batch size"
            failed_models.append((config.name, error_msg))

    if failed_models:
        print("\n\nSummary of failed models:")
        print("========================")
        for model_name, error in failed_models:
            print(f"{model_name}: {error}")
    else:
        print("\n\nAll models trained successfully!")


if __name__ == "__main__":
    train_models()
