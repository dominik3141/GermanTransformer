"""
Train the model.
"""

import configparser
import torch
from torch import nn
from src.nn import TransformerEncoder
from src.nouns import load_nouns_from_csv, create_train_val_dataloaders
import wandb
import time

config = configparser.ConfigParser()
config.read("default.conf")


def train(test: bool = False, debug: bool = False):
    """Trains the transformer model on German noun article classification"""

    # Model parameters
    model_params = {
        "max_sequence_length": int(config["MODEL"]["max_sequence_length"]),
        "d_model": int(config["MODEL"]["d_model"]),
        "num_heads": int(config["MODEL"]["num_heads"]),
        "num_layers": int(config["MODEL"]["num_layers"]),
        "num_classes": 3,  # m, f, n
    }

    # Training parameters
    batch_size = int(config["TRAINING"]["batch_size"])
    learning_rate = float(config["TRAINING"]["learning_rate"])
    weight_decay = float(config["TRAINING"]["weight_decay"])
    epochs = int(config["TRAINING"]["epochs"])
    dropout_rate = float(config["TRAINING"]["dropout_rate"])

    # Load and split data
    nouns = load_nouns_from_csv(config["DATA"]["nouns_path"])
    train_loader, val_loader = create_train_val_dataloaders(
        nouns, batch_size=batch_size
    )

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = TransformerEncoder(**model_params, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if not test:
        wandb.init(
            project="german-articles",
            config={
                **model_params,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "dropout_rate": dropout_rate,
            },
        )

        # Log all model parameters every 50 steps
        wandb.watch(model, log="all", log_freq=50)

    # Add a global step counter at the start of training
    global_step = 0

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Add timing metrics
        batch_times = []
        epoch_start = time.time()
        batch_start = time.time()
        batch_count = 0

        for words, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(words)
            loss = criterion(outputs, labels)

            # the model already outputs probabilities
            probabilities = outputs

            # Calculate prediction confidence and diversity
            batch_confidence = torch.mean(torch.max(probabilities, dim=1)[0]).item()

            # Calculate prediction diversity (std dev across batch for each class)
            pred_diversity = torch.std(probabilities, dim=0).mean().item()

            if debug:
                random_idx = torch.randint(0, len(words), (1,)).item()
                probs = outputs[random_idx].detach().cpu().numpy()
                word = words[random_idx]
                true_label = labels[random_idx].item()
                print("\nDebug - Random sample:")
                print(f"Word: {word}")
                print(f"True label: {['m', 'f', 'n'][true_label]}")
                print("Probabilities:")
                print(f"  masculine: {probs[0]:.4f}")
                print(f"  feminine: {probs[1]:.4f}")
                print(f"  neuter: {probs[2]:.4f}")
                print("-" * 40)

            loss.backward()

            if not test:
                # Log training metrics with explicit step
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/prediction_confidence": batch_confidence,
                        "train/prediction_diversity": pred_diversity,
                    },
                    step=global_step,
                )

            if test:
                print(f"Train Loss: {loss.item():.4f}")
                print(f"Batch confidence: {batch_confidence:.4f}")
                print(f"Prediction diversity: {pred_diversity:.4f}")

            optimizer.step()

            # Calculate throughput every 20 batches
            batch_count += 1
            if batch_count % 20 == 0:
                batch_end = time.time()
                time_for_20_batches = batch_end - batch_start
                batch_times.append(
                    time_for_20_batches / 20
                )  # Add average time per batch

                if not test:
                    wandb.log(
                        {
                            "performance/batches_per_minute": (20 * 60.0)
                            / time_for_20_batches,
                            "performance/samples_per_minute": (20 * batch_size * 60.0)
                            / time_for_20_batches,
                        },
                        step=global_step,
                    )
                batch_start = time.time()  # Reset timer for next 20 batches

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            # Evaluate Nutella every 50 batches
            if batch_count % 50 == 0:
                model.eval()
                with torch.no_grad():
                    nutella_word = ["Nutella"]
                    nutella_output = model(nutella_word)
                    nutella_probs = nutella_output[0]

                    nutella_metrics = {
                        "nutella/masculine_prob": nutella_probs[0].item(),
                        "nutella/feminine_prob": nutella_probs[1].item(),
                        "nutella/neuter_prob": nutella_probs[2].item(),
                    }

                    if not test:
                        wandb.log(nutella_metrics, step=global_step)
                    else:
                        print("\nNutella Predictions:")
                        print(f"Masculine (der): {nutella_probs[0]:.4f}")
                        print(f"Feminine (die): {nutella_probs[1]:.4f}")
                        print(f"Neuter (das): {nutella_probs[2]:.4f}")
                model.train()  # Switch back to training mode

            # Update step counter after each batch
            global_step += 1

        # Calculate epoch-level performance metrics
        epoch_time = time.time() - epoch_start
        avg_batch_time = sum(batch_times) / len(batch_times)

        performance_metrics = {
            "performance/epoch_time": epoch_time,
            "performance/avg_batches_per_minute": 60.0 / avg_batch_time,
            "performance/avg_samples_per_minute": (batch_size * 60.0) / avg_batch_time,
        }

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        # Continue with existing validation loop
        with torch.no_grad():
            for words, labels in val_loader:
                labels = labels.to(device)
                outputs = model(words)
                loss = criterion(outputs, labels)

                # Calculate validation confidence and diversity
                probabilities = torch.softmax(outputs, dim=1)
                batch_confidence = torch.mean(torch.max(probabilities, dim=1)[0]).item()
                pred_diversity = torch.std(probabilities, dim=0).mean().item()

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                if not test:
                    wandb.log(
                        {
                            "val/prediction_confidence": batch_confidence,
                            "val/prediction_diversity": pred_diversity,
                        },
                        step=global_step,  # Use same step as training
                    )

        # Print and log epoch statistics
        train_metrics = {
            "train/loss": train_loss / len(train_loader),
            "train/accuracy": 100.0 * train_correct / train_total,
            "val/loss": val_loss / len(val_loader),
            "val/accuracy": 100.0 * val_correct / val_total,
            "epoch": epoch + 1,
        }

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(
            f"Train Loss: {train_metrics['train/loss']:.4f}, "
            f"Train Acc: {train_metrics['train/accuracy']:.2f}%"
        )
        print(
            f"Val Loss: {train_metrics['val/loss']:.4f}, "
            f"Val Acc: {train_metrics['val/accuracy']:.2f}%"
        )

        if not test:
            wandb.log({**train_metrics, **performance_metrics}, step=global_step)
            # Save model with wandb
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")
            wandb.save(f"checkpoints/model_epoch_{epoch}.pth")

        # if test, break after first epoch
        if test:
            break

    if not test:
        wandb.finish()


if __name__ == "__main__":
    train(debug=True)
