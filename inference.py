import os
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

from models.siamese import SiameseNetwork
from models.meta_learner import MAML


def load_model(model_path, config_path, device):
    """
    Load a trained model (either Siamese or Meta)

    Args:
        model_path: Path to the model checkpoint
        config_path: Path to configuration
        device: Device to load the model on

    Returns:
        The loaded model
    """
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Create siamese model
    siamese_model = SiameseNetwork(
        backbone=config["siamese"]["backbone"],
        embedding_dim=config["siamese"]["embedding_dim"],
        pretrained=False,
    ).to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    siamese_model.load_state_dict(checkpoint["model_state_dict"])
    siamese_model.eval()

    return siamese_model


def preprocess_image(image_path, config):
    """
    Preprocess an image for inference

    Args:
        image_path: Path to image
        config: Configuration dictionary

    Returns:
        Preprocessed tensor
    """
    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize(config["data"]["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return tensor


def compute_similarity(model, img1_tensor, img2_tensor, device):
    """
    Compute similarity between two images using the model

    Args:
        model: Trained Siamese model
        img1_tensor: First image tensor
        img2_tensor: Second image tensor
        device: Device to run inference on

    Returns:
        Similarity score (0-1), where 1 means identical and 0 means different
    """
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    with torch.no_grad():
        # Get embeddings
        emb1, emb2 = model(img1_tensor, img2_tensor)

        # Compute distance
        distance = model.compute_distance(emb1, emb2)

        # Convert distance to similarity (1 - distance)
        similarity = 1.0 - distance

    return similarity.item()


def adapt_to_support_set(model, support_images, support_pairs, device, config):
    """
    Adapt the model to a support set of images using meta-learning

    Args:
        model: Base Siamese model
        support_images: List of support image paths
        support_pairs: List of tuples (idx1, idx2, label) defining pairs
        device: Device to run adaptation on
        config: Configuration dictionary

    Returns:
        Adapted model
    """
    # Create a copy of the model for adaptation
    adapted_model = SiameseNetwork(
        backbone=config["siamese"]["backbone"],
        embedding_dim=config["siamese"]["embedding_dim"],
        pretrained=False,
    ).to(device)
    adapted_model.load_state_dict(model.state_dict())

    # Wrap with MAML
    meta_model = MAML(
        model=adapted_model, inner_lr=config["meta"]["inner_lr"], first_order=True
    ).to(device)

    # Load and preprocess support images
    support_tensors = []
    for img_path in support_images:
        tensor = preprocess_image(img_path, config)
        support_tensors.append(tensor)

    # Create support pairs
    if not support_pairs:
        # If no pairs are specified, create all possible pairs assuming similarity
        support_x1 = []
        support_x2 = []
        support_y = []

        for i in range(len(support_tensors)):
            for j in range(i + 1, len(support_tensors)):
                support_x1.append(support_tensors[i])
                support_x2.append(support_tensors[j])
                support_y.append(torch.tensor(1.0))  # Assume all are similar
    else:
        # Create pairs from the specified tuples
        support_x1 = []
        support_x2 = []
        support_y = []

        for idx1, idx2, label in support_pairs:
            support_x1.append(support_tensors[idx1])
            support_x2.append(support_tensors[idx2])
            support_y.append(torch.tensor(float(label)))

    # Skip adaptation if no pairs
    if len(support_x1) == 0:
        return adapted_model

    # Convert to tensors
    support_x1 = torch.cat(support_x1).to(device)
    support_x2 = torch.cat(support_x2).to(device)
    support_y = torch.stack(support_y).to(device)

    # Create optimizer for adaptation
    optimizer = torch.optim.SGD(
        adapted_model.parameters(), lr=config["meta"]["inner_lr"]
    )

    # Adaptation steps
    for step in range(config["meta"]["adapt_steps"]):
        # Forward pass
        emb1, emb2 = adapted_model(support_x1, support_x2)
        distance = adapted_model.compute_distance(emb1, emb2)

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(1.0 - distance, support_y)

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Adaptation step {step + 1}, Loss: {loss.item():.4f}")

    return adapted_model


def visualize_similarity(img1_path, img2_path, similarity):
    """
    Visualize two images side by side with similarity score
    """
    plt.figure(figsize=(10, 5))

    # First image
    plt.subplot(1, 2, 1)
    img1 = plt.imread(img1_path)
    plt.imshow(img1, cmap="gray")
    plt.title("Image 1")
    plt.axis("off")

    # Second image
    plt.subplot(1, 2, 2)
    img2 = plt.imread(img2_path)
    plt.imshow(img2, cmap="gray")
    plt.title(f"Image 2 - Similarity: {similarity:.4f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("similarity_visualization.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with trained models")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config"
    )
    parser.add_argument(
        "--mode",
        choices=["similarity", "adaptation"],
        default="similarity",
        help="Inference mode: compute similarity or adapt to support set",
    )
    parser.add_argument(
        "--img1", type=str, help="Path to first image (similarity mode)"
    )
    parser.add_argument(
        "--img2", type=str, help="Path to second image (similarity mode)"
    )
    parser.add_argument(
        "--support_dir",
        type=str,
        help="Directory with support images (adaptation mode)",
    )
    parser.add_argument(
        "--query_dir", type=str, help="Directory with query images (adaptation mode)"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Load model
    model = load_model(args.model_path, args.config, device)
    print(f"Loaded model from {args.model_path}")

    if args.mode == "similarity":
        # Check if both images are provided
        if not args.img1 or not args.img2:
            raise ValueError("Both --img1 and --img2 are required in similarity mode")

        # Process images
        img1_tensor = preprocess_image(args.img1, config)
        img2_tensor = preprocess_image(args.img2, config)

        # Compute similarity
        similarity = compute_similarity(model, img1_tensor, img2_tensor, device)
        print(f"Similarity between images: {similarity:.4f}")

        # Visualize
        visualize_similarity(args.img1, args.img2, similarity)

    elif args.mode == "adaptation":
        # Check if support directory is provided
        if not args.support_dir:
            raise ValueError("--support_dir is required in adaptation mode")

        # Get support images
        support_dir = Path(args.support_dir)
        support_images = list(support_dir.glob("*.png")) + list(
            support_dir.glob("*.jpg")
        )
        support_images = [str(path) for path in support_images]

        if len(support_images) == 0:
            raise ValueError(f"No images found in support directory {args.support_dir}")

        print(f"Found {len(support_images)} support images")

        # Adapt model
        adapted_model = adapt_to_support_set(model, support_images, [], device, config)
        print("Model adapted to support set")

        # If query directory is provided, evaluate on query images
        if args.query_dir:
            query_dir = Path(args.query_dir)
            query_images = list(query_dir.glob("*.png")) + list(query_dir.glob("*.jpg"))
            query_images = [str(path) for path in query_images]

            if len(query_images) == 0:
                print(f"No images found in query directory {args.query_dir}")
            else:
                print(f"Evaluating on {len(query_images)} query images...")

                # Compare each query image with the first support image
                reference_img = support_images[0]
                reference_tensor = preprocess_image(reference_img, config)

                # Print results for standard model and adapted model
                for i, query_img in enumerate(query_images):
                    query_tensor = preprocess_image(query_img, config)

                    # Similarity with standard model
                    standard_similarity = compute_similarity(
                        model, reference_tensor, query_tensor, device
                    )

                    # Similarity with adapted model
                    adapted_similarity = compute_similarity(
                        adapted_model, reference_tensor, query_tensor, device
                    )

                    print(f"Query image {i + 1}:")
                    print(f"  Standard similarity: {standard_similarity:.4f}")
                    print(f"  Adapted similarity: {adapted_similarity:.4f}")
                    print(
                        f"  Improvement: {adapted_similarity - standard_similarity:.4f}"
                    )
                    print()
