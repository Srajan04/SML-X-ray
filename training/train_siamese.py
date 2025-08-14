import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import math

# Add the project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.siamese import SiameseNetwork
from losses.contrastive_loss import ContrastiveLoss
from data.dataloader import get_data_loaders


def train_siamese_network(config_path="config/config.yaml"):
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader = get_data_loaders(config)

    # Create model
    model = SiameseNetwork(
        backbone=config["siamese"]["backbone"],
        embedding_dim=config["siamese"]["embedding_dim"],
        pretrained=config["siamese"]["pretrained"],
    ).to(device)

    # Define loss function
    criterion = ContrastiveLoss(margin=config["siamese"]["margin"])

    # Define optimizer
    if config["optimizer"]["name"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["optimizer"]["lr"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["optimizer"]["lr"],
            momentum=0.9,
            weight_decay=config["optimizer"]["weight_decay"],
        )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    # Training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    # Create directory for saving checkpoints
    save_dir = os.path.join(os.getcwd(), config["training"]["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    # Training process
    for epoch in range(config["training"]["siamese_epochs"]):
        # Training phase
        model.train()
        running_loss = 0.0

        for batch_idx, (img1, img2, target) in enumerate(
            tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{config['training']['siamese_epochs']}",
            )
        ):
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            embedding1, embedding2 = model(img1, img2)
            distance = model.compute_distance(
                embedding1, embedding2, metric=config["siamese"]["distance_metric"]
            )

            # Compute loss
            loss = criterion(distance, target)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % config["training"]["log_interval"] == 0:
                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for img1, img2, target in tqdm(val_loader, desc="Validation"):
                img1, img2, target = img1.to(device), img2.to(device), target.to(device)

                # Forward pass
                embedding1, embedding2 = model(img1, img2)
                distance = model.compute_distance(
                    embedding1, embedding2, metric=config["siamese"]["distance_metric"]
                )

                # Compute loss
                loss = criterion(distance, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                os.path.join(save_dir, "best_siamese_model.pth"),
            )
            print(f"Saved best model with validation loss: {val_loss:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= config["training"]["early_stopping_patience"]:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # Save checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            },
            os.path.join(save_dir, f"siamese_epoch_{epoch + 1}.pth"),
        )

    # Plot and save training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.savefig(os.path.join(save_dir, "siamese_training_curve.png"))
    plt.close()

    print("Training completed!")
    return model


if __name__ == "__main__":
    train_siamese_network()
