import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import os
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE


def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_score, save_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def visualize_embeddings(embeddings, labels, save_path):
    """Visualize embeddings using t-SNE"""
    # Apply t-SNE
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1)
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 8))

    # Convert labels to integers if they're one-hot encoded
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        # Convert multi-label to single color (based on first disease)
        plot_labels = labels[:, 0].astype(int)
    else:
        plot_labels = labels

    unique_labels = np.unique(plot_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = plot_labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=f"Class {label}",
            alpha=0.7,
            edgecolors="w",
        )

    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def visualize_image_pairs(img1, img2, similarity, save_path):
    """Visualize pairs of images with their similarity scores"""
    # Number of pairs to visualize
    n_pairs = min(len(img1), 5)

    plt.figure(figsize=(12, 4 * n_pairs))

    for i in range(n_pairs):
        # Display first image
        plt.subplot(n_pairs, 2, i * 2 + 1)
        img = img1[i].cpu().permute(1, 2, 0).numpy()
        # Denormalize if needed
        img = np.clip((img * 0.229 + 0.485) * 255, 0, 255).astype(np.uint8)
        plt.imshow(img)
        plt.title(f"Image 1 - Pair {i + 1}")
        plt.axis("off")

        # Display second image
        plt.subplot(n_pairs, 2, i * 2 + 2)
        img = img2[i].cpu().permute(1, 2, 0).numpy()
        # Denormalize if needed
        img = np.clip((img * 0.229 + 0.485) * 255, 0, 255).astype(np.uint8)
        plt.imshow(img)
        plt.title(f"Image 2 - Similarity: {1 - similarity[i]:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_adaptation_progress(distances_before, distances_after, labels, save_path):
    """Visualize improvement after adaptation"""
    plt.figure(figsize=(10, 6))

    # Sort by true similarity for better visualization
    indices = np.argsort(labels)
    sorted_labels = labels[indices]
    sorted_before = distances_before[indices]
    sorted_after = distances_after[indices]

    # Plot distances before and after adaptation
    width = 0.35
    x = np.arange(len(sorted_labels))
    plt.bar(x - width / 2, 1 - sorted_before, width, label="Before Adaptation")
    plt.bar(x + width / 2, 1 - sorted_after, width, label="After Adaptation")

    # Add scatter plot of true labels
    plt.scatter(x, sorted_labels, color="red", marker="x", s=50, label="True Label")

    plt.xlabel("Pair Index")
    plt.ylabel("Similarity Score (1 - Distance)")
    plt.title("Similarity Scores Before and After Adaptation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
