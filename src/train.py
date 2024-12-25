"""
Train the model.
"""

import configparser
import torch
from torch import nn
from src.nn import TransformerEncoder
from src.nouns import load_nouns_from_csv, create_train_val_dataloaders

config = configparser.ConfigParser()
config.read("default.conf")


def train(test: bool = False):
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

    # Load and split data
    nouns = load_nouns_from_csv(config["DATA"]["nouns_path"])
    train_loader, val_loader = create_train_val_dataloaders(
        nouns, batch_size=batch_size
    )

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerEncoder(**model_params).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for words, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(words)
            loss = criterion(outputs, labels)

            loss.backward()

            # print the loss
            print(f"Loss: {loss.item()}")
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for words, labels in val_loader:
                labels = labels.to(device)
                outputs = model(words)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Print epoch statistics
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(
            f"Train Loss: {train_loss/len(train_loader):.4f}, "
            f"Train Acc: {100.*train_correct/train_total:.2f}%"
        )
        print(
            f"Val Loss: {val_loss/len(val_loader):.4f}, "
            f"Val Acc: {100.*val_correct/val_total:.2f}%"
        )

        # if test, break after first epoch
        if test:
            break

        if not test:
            # Save model
            torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")
