import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
import sys
import pandas as pd

# Add the project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.siamese import SiameseNetwork
from models.meta_learner import MAML
from data.dataloader import CheXpertDataset, SiamesePairDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# Add this function to the file


def evaluate_similarity_model(
    model_path, config_path="config/config.yaml", mode="siamese", adaptation_steps=3
):
    """
    Evaluate the model on image similarity tasks across patients
    Modified to work with CheXpert's single image per patient pattern
    """
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    siamese_model = SiameseNetwork(
        backbone=config["siamese"]["backbone"],
        embedding_dim=config["siamese"]["embedding_dim"],
        pretrained=False,
    ).to(device)

    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    siamese_model.load_state_dict(checkpoint["model_state_dict"])
    siamese_model.eval()

    # Create test dataset
    test_transform = transforms.Compose(
        [
            transforms.Resize(config["data"]["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_csv = os.path.join(config["data"]["dataset_path"], config["data"]["test_csv"])
    test_dataset = CheXpertDataset(
        test_csv, config["data"]["dataset_path"], transform=test_transform
    )

    # Instead of evaluating per patient, we'll evaluate by disease
    disease_pairs = []
    disease_indices = {}

    # Organize images by disease
    for disease_idx, disease_name in enumerate(test_dataset.target_diseases):
        pos_indices = []
        neg_indices = []

        # Find positive and negative cases for each disease
        for i in range(len(test_dataset)):
            img, label, patient_id = test_dataset[i]
            if label[disease_idx] == 1:
                pos_indices.append(i)
            elif label[disease_idx] == 0:
                neg_indices.append(i)

        # Store the indices by disease
        disease_indices[disease_name] = {
            "positive": pos_indices,
            "negative": neg_indices,
        }

        print(
            f"Disease {disease_name}: {len(pos_indices)} positive, {len(neg_indices)} negative"
        )

    # Evaluation metrics
    disease_accuracies = []
    disease_similarities = []

    # For each disease with enough samples, create evaluation pairs
    for disease_name, indices in disease_indices.items():
        pos_indices = indices["positive"]
        neg_indices = indices["negative"]

        # Skip diseases with too few samples
        if len(pos_indices) < 3 or len(neg_indices) < 3:
            print(f"Skipping {disease_name}: not enough samples")
            continue

        print(f"\nEvaluating {disease_name}...")

        # Create pairs for support and query
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        # Use a few samples for adaptation in meta-learning
        support_pos = pos_indices[: min(3, len(pos_indices) // 2)]
        support_neg = neg_indices[: min(3, len(neg_indices) // 2)]

        query_pos = pos_indices[min(3, len(pos_indices) // 2) :]
        query_neg = neg_indices[min(3, len(neg_indices) // 2) :]

        # Create balanced support pairs
        support_pairs = []
        # Similar pairs (pos-pos)
        for i in range(len(support_pos)):
            for j in range(i + 1, len(support_pos)):
                img1, _, _ = test_dataset[support_pos[i]]
                img2, _, _ = test_dataset[support_pos[j]]
                support_pairs.append((img1, img2, 1.0))  # Similar

        # Similar pairs (neg-neg)
        for i in range(len(support_neg)):
            for j in range(i + 1, len(support_neg)):
                img1, _, _ = test_dataset[support_neg[i]]
                img2, _, _ = test_dataset[support_neg[j]]
                support_pairs.append((img1, img2, 1.0))  # Similar

        # Dissimilar pairs (pos-neg)
        for i in range(min(len(support_pos), len(support_neg))):
            img1, _, _ = test_dataset[support_pos[i]]
            img2, _, _ = test_dataset[support_neg[i]]
            support_pairs.append((img1, img2, 0.0))  # Dissimilar

        # Create balanced query pairs
        query_pairs = []
        # Similar pairs (pos-pos)
        for i in range(min(10, len(query_pos))):
            for j in range(i + 1, min(10, len(query_pos))):
                img1, _, _ = test_dataset[query_pos[i]]
                img2, _, _ = test_dataset[query_pos[j]]
                query_pairs.append((img1, img2, 1.0))  # Similar

        # Similar pairs (neg-neg)
        for i in range(min(10, len(query_neg))):
            for j in range(i + 1, min(10, len(query_neg))):
                img1, _, _ = test_dataset[query_neg[i]]
                img2, _, _ = test_dataset[query_neg[j]]
                query_pairs.append((img1, img2, 1.0))  # Similar

        # Dissimilar pairs (pos-neg)
        for i in range(min(10, min(len(query_pos), len(query_neg)))):
            img1, _, _ = test_dataset[query_pos[i]]
            img2, _, _ = test_dataset[query_neg[i]]
            query_pairs.append((img1, img2, 0.0))  # Dissimilar

        # Skip if not enough pairs
        if len(support_pairs) < 3 or len(query_pairs) < 3:
            print(f"Skipping {disease_name}: not enough valid pairs")
            continue

        # If meta mode, adapt on support pairs
        if mode == "meta":
            print(f"Adapting model on {len(support_pairs)} support pairs...")
            # Adapt the model
            support_x1 = torch.stack([p[0] for p in support_pairs]).to(device)
            support_x2 = torch.stack([p[1] for p in support_pairs]).to(device)
            support_y = torch.tensor([p[2] for p in support_pairs]).to(device)

            # Create a copy for adaptation
            adapted_model = SiameseNetwork(
                backbone=config["siamese"]["backbone"],
                embedding_dim=config["siamese"]["embedding_dim"],
                pretrained=False,
            ).to(device)
            adapted_model.load_state_dict(siamese_model.state_dict())

            # Adaptation loop
            optimizer = torch.optim.SGD(adapted_model.parameters(), lr=0.01)
            for _ in range(adaptation_steps):
                emb1, emb2 = adapted_model(support_x1, support_x2)
                distance = adapted_model.compute_distance(emb1, emb2)
                loss = F.binary_cross_entropy_with_logits(
                    1.0 - distance, support_y.float()
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate adapted model
            model_to_eval = adapted_model
        else:
            # Use standard model without adaptation
            model_to_eval = siamese_model

        # Evaluate on query pairs
        print(f"Evaluating on {len(query_pairs)} query pairs...")
        test_x1 = torch.stack([p[0] for p in query_pairs]).to(device)
        test_x2 = torch.stack([p[1] for p in query_pairs]).to(device)
        test_y = torch.tensor([p[2] for p in query_pairs]).to(device)

        # Get predictions
        with torch.no_grad():
            emb1, emb2 = model_to_eval(test_x1, test_x2)
            distance = model_to_eval.compute_distance(emb1, emb2)
            similarity = 1.0 - distance

        # Calculate metrics
        predictions = (similarity > 0.5).float().cpu()
        accuracy = (predictions == test_y.cpu()).float().mean().item()
        avg_similarity = similarity.mean().item()

        disease_accuracies.append(accuracy)
        disease_similarities.append(avg_similarity)

        print(
            f"Disease {disease_name}: Accuracy = {accuracy:.4f}, Avg Similarity = {avg_similarity:.4f}"
        )

    # Print overall results
    if len(disease_accuracies) > 0:
        print(f"\nEvaluation completed on {len(disease_accuracies)} diseases")
        print(f"Average accuracy: {np.mean(disease_accuracies):.4f}")
        print(f"Average similarity score: {np.mean(disease_similarities):.4f}")
    else:
        print("No valid diseases found for evaluation")

    return disease_accuracies, disease_similarities


def evaluate_model(
    model_path, config_path="config/config.yaml", mode="siamese", adaptation_steps=5
):
    """
    Evaluate the model on the test set

    Args:
        model_path: Path to the saved model
        config_path: Path to the config file
        mode: 'siamese' or 'meta' evaluation mode
        adaptation_steps: Number of adaptation steps for meta-learning
    """
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    siamese_model = SiameseNetwork(
        backbone=config["siamese"]["backbone"],
        embedding_dim=config["siamese"]["embedding_dim"],
        pretrained=False,
    ).to(device)

    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    siamese_model.load_state_dict(checkpoint["model_state_dict"])
    siamese_model.eval()

    # If meta-learning evaluation, wrap with MAML
    if mode == "meta":
        meta_model = MAML(
            model=siamese_model, inner_lr=config["meta"]["inner_lr"], first_order=True
        ).to(device)

    # Create test dataset
    test_transform = transforms.Compose(
        [
            transforms.Resize(config["data"]["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_csv = os.path.join(config["data"]["dataset_path"], config["data"]["test_csv"])
    test_dataset = CheXpertDataset(
        test_csv, config["data"]["dataset_path"], transform=test_transform
    )

    # Group by patient for few-shot adaptation testing
    patients = {}
    for i in range(len(test_dataset)):
        _, label, patient_id = test_dataset[i]
        if patient_id not in patients:
            patients[patient_id] = []
        patients[patient_id].append(i)

    # Evaluation metrics
    patient_aucs = []
    patient_aps = []

    # For each patient, perform few-shot adaptation and evaluate
    for patient_id, indices in tqdm(patients.items(), desc="Evaluating patients"):
        if len(indices) < 10:  # Skip patients with too few samples
            continue

        # Split indices into support (adaptation) and query (evaluation) sets
        np.random.shuffle(indices)
        support_indices = indices[: min(config["meta"]["shots"] * 2, len(indices) // 2)]
        query_indices = indices[min(config["meta"]["shots"] * 2, len(indices) // 2) :]

        if len(query_indices) < 5:  # Skip if not enough samples for evaluation
            continue

        # Create pairs for support set
        support_pairs = []
        for i in range(len(support_indices)):
            for j in range(i + 1, len(support_indices)):
                idx1, idx2 = support_indices[i], support_indices[j]
                img1, label1, _ = test_dataset[idx1]
                img2, label2, _ = test_dataset[idx2]
                is_same = torch.all(torch.eq(label1, label2))
                support_pairs.append(
                    (img1, img2, torch.tensor(1.0) if is_same else torch.tensor(0.0))
                )

        # Create pairs for query set
        query_pairs = []
        query_labels = []
        for i in range(len(query_indices)):
            for j in range(i + 1, len(query_indices)):
                idx1, idx2 = query_indices[i], query_indices[j]
                img1, label1, _ = test_dataset[idx1]
                img2, label2, _ = test_dataset[idx2]
                is_same = torch.all(torch.eq(label1, label2))
                query_pairs.append((img1, img2))
                query_labels.append(1.0 if is_same else 0.0)

        # Convert to tensors
        query_labels = torch.tensor(query_labels)

        if mode == "meta" and len(support_pairs) > 0:
            # Adapt the model to this patient's data
            adapted_model = adapt_model(
                meta_model, support_pairs, device, steps=adaptation_steps
            )
            # Evaluate the adapted model
            predictions = evaluate_pairs(adapted_model, query_pairs, device)
        else:
            # Standard Siamese evaluation without adaptation
            predictions = evaluate_pairs(siamese_model, query_pairs, device)

        # Calculate metrics
        if len(np.unique(query_labels.numpy())) > 1:  # Only if both classes present
            fpr, tpr, _ = roc_curve(query_labels.numpy(), 1 - predictions)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(
                query_labels.numpy(), 1 - predictions
            )
            ap = average_precision_score(query_labels.numpy(), 1 - predictions)

            patient_aucs.append(roc_auc)
            patient_aps.append(ap)

            print(f"Patient {patient_id}: ROC AUC = {roc_auc:.4f}, AP = {ap:.4f}")

    # Print average metrics
    if patient_aucs:
        print(
            f"Average ROC AUC: {np.mean(patient_aucs):.4f} ± {np.std(patient_aucs):.4f}"
        )
        print(f"Average AP: {np.mean(patient_aps):.4f} ± {np.std(patient_aps):.4f}")

    return patient_aucs, patient_aps


def adapt_model(meta_model, support_pairs, device, steps=5):
    """Adapt the model to the support set"""
    # Copy the model for adaptation
    adapted_model = meta_model.model

    if len(support_pairs) == 0:
        return adapted_model

    # Extract data from support pairs
    support_x1 = torch.stack([pair[0] for pair in support_pairs]).to(device)
    support_x2 = torch.stack([pair[1] for pair in support_pairs]).to(device)
    support_y = torch.stack([pair[2] for pair in support_pairs]).to(device)

    # Create optimizer for adaptation
    optimizer = torch.optim.SGD(adapted_model.parameters(), lr=0.01)

    # Adaptation steps
    for _ in range(steps):
        # Forward pass
        emb1, emb2 = adapted_model(support_x1, support_x2)
        distance = adapted_model.compute_distance(emb1, emb2)

        # Compute loss - using BCE directly as a simple approach
        loss = F.binary_cross_entropy_with_logits(1.0 - distance, support_y.float())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return adapted_model


def evaluate_pairs(model, pairs, device):
    """Evaluate model on image pairs and return similarity scores"""
    model.eval()
    predictions = []

    with torch.no_grad():
        for pair in pairs:
            img1, img2 = pair
            img1 = img1.unsqueeze(0).to(device)
            img2 = img2.unsqueeze(0).to(device)

            # Forward pass
            emb1, emb2 = model(img1, img2)
            distance = model.compute_distance(emb1, emb2)

            predictions.append(distance.cpu().item())

    return np.array(predictions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate Siamese or Meta-Learning model"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the saved model"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="siamese",
        choices=["siamese", "meta"],
        help="Evaluation mode: siamese or meta",
    )
    parser.add_argument(
        "--adapt_steps",
        type=int,
        default=5,
        help="Number of adaptation steps for meta-learning",
    )

    args = parser.parse_args()

    evaluate_model(args.model_path, args.config, args.mode, args.adapt_steps)
