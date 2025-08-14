import os
import torch
import yaml
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.siamese import SiameseNetwork
from data.dataloader import CheXpertDataset, CachedDataset


def predict_with_meta_adaptation(
    config_path="config/config.yaml", model_path=None, support_size=10
):
    """
    Use meta-learning model to predict diseases with adaptation

    Args:
        config_path: Path to configuration file
        model_path: Path to trained meta model (if None, use best_meta_model.pth)
        support_size: Number of examples to use for adaptation
    """
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if model_path is None:
        model_path = os.path.join(config["training"]["save_dir"], "best_meta_model.pth")

    # Create model
    siamese_model = SiameseNetwork(
        backbone=config["siamese"]["backbone"],
        embedding_dim=config["siamese"]["embedding_dim"],
        pretrained=False,
    ).to(device)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    siamese_model.load_state_dict(checkpoint["model_state_dict"])
    siamese_model.eval()

    # Load test data
    use_cache = config["data"].get("use_cache", False)
    cache_dir = config["data"].get("cache_dir", None)

    if (
        use_cache
        and cache_dir
        and os.path.exists(os.path.join(cache_dir, "valid_cache.csv"))
    ):
        print("Using cached test dataset")
        test_dataset = CachedDataset(os.path.join(cache_dir, "valid_cache.csv"))
    else:
        print("Using regular test dataset")
        transform = transforms.Compose(
            [
                transforms.Resize(config["data"]["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        test_csv = os.path.join(
            config["data"]["dataset_path"], config["data"]["valid_csv"]
        )
        test_dataset = CheXpertDataset(
            test_csv, config["data"]["dataset_path"], transform=transform
        )

    # Get disease names
    disease_names = test_dataset.target_diseases
    print(f"Diseases to predict: {disease_names}")

    # We'll evaluate each disease separately
    results = {}

    for disease_idx, disease_name in enumerate(disease_names):
        print(f"\nEvaluating disease: {disease_name}")

        # Find positive and negative cases
        pos_indices = []
        neg_indices = []

        for i in range(len(test_dataset)):
            _, label, _ = test_dataset[i]
            if label[disease_idx] == 1:
                pos_indices.append(i)
            elif label[disease_idx] == 0:
                neg_indices.append(i)

        print(
            f"Found {len(pos_indices)} positive and {len(neg_indices)} negative cases"
        )

        if len(pos_indices) < 5 or len(neg_indices) < 5:
            print(f"Skipping {disease_name} due to insufficient samples")
            continue

        # Split into support and query sets
        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        support_pos = pos_indices[: min(support_size // 2, len(pos_indices) // 2)]
        support_neg = neg_indices[: min(support_size // 2, len(neg_indices) // 2)]

        query_pos = pos_indices[min(support_size // 2, len(pos_indices) // 2) :]
        query_neg = neg_indices[min(support_size // 2, len(neg_indices) // 2) :]

        # Create support pairs for adaptation
        support_pairs = []

        # Similar positive pairs
        for i in range(len(support_pos)):
            for j in range(i + 1, len(support_pos)):
                img1, _, _ = test_dataset[support_pos[i]]
                img2, _, _ = test_dataset[support_pos[j]]
                support_pairs.append((img1, img2, torch.tensor(1.0)))

        # Similar negative pairs
        for i in range(len(support_neg)):
            for j in range(i + 1, len(support_neg)):
                img1, _, _ = test_dataset[support_neg[i]]
                img2, _, _ = test_dataset[support_neg[j]]
                support_pairs.append((img1, img2, torch.tensor(1.0)))

        # Dissimilar pairs
        for i in range(min(len(support_pos), len(support_neg))):
            img1, _, _ = test_dataset[support_pos[i]]
            img2, _, _ = test_dataset[support_neg[i]]
            support_pairs.append((img1, img2, torch.tensor(0.0)))

        # Skip if not enough pairs for adaptation
        if len(support_pairs) < 3:
            print(f"Skipping {disease_name}: not enough pairs for adaptation")
            continue

        # Adapt the model using support pairs
        print(f"Adapting model on {len(support_pairs)} support pairs...")

        # Create a copy of the model for adaptation
        adapted_model = SiameseNetwork(
            backbone=config["siamese"]["backbone"],
            embedding_dim=config["siamese"]["embedding_dim"],
            pretrained=False,
        ).to(device)
        adapted_model.load_state_dict(siamese_model.state_dict())
        adapted_model.train()

        # Prepare data for adaptation
        support_x1 = torch.stack([p[0] for p in support_pairs]).to(device)
        support_x2 = torch.stack([p[1] for p in support_pairs]).to(device)
        support_y = torch.stack([p[2] for p in support_pairs]).to(device)

        # Adaptation loop
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=0.01)
        for _ in range(config["meta"]["adapt_steps"]):
            emb1, emb2 = adapted_model(support_x1, support_x2)
            distance = adapted_model.compute_distance(emb1, emb2)
            loss = F.binary_cross_entropy_with_logits(1.0 - distance, support_y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # After adaptation, use adapted model for prediction
        # Create a reference embedding for positive and negative
        adapted_model.eval()

        with torch.no_grad():
            # Create reference embeddings by averaging
            pos_ref_embedding = torch.zeros(1, config["siamese"]["embedding_dim"]).to(
                device
            )
            neg_ref_embedding = torch.zeros(1, config["siamese"]["embedding_dim"]).to(
                device
            )

            # Average positive embeddings
            for idx in support_pos:
                img, _, _ = test_dataset[idx]
                img = img.unsqueeze(0).to(device)
                embedding = adapted_model.forward_one(img)
                pos_ref_embedding += embedding
            pos_ref_embedding /= len(support_pos)

            # Average negative embeddings
            for idx in support_neg:
                img, _, _ = test_dataset[idx]
                img = img.unsqueeze(0).to(device)
                embedding = adapted_model.forward_one(img)
                neg_ref_embedding += embedding
            neg_ref_embedding /= len(support_neg)

            # Evaluate on query set
            predictions = []
            true_labels = []

            for idx in tqdm(query_pos, desc="Evaluating positive cases"):
                img, _, _ = test_dataset[idx]
                img = img.unsqueeze(0).to(device)
                embedding = adapted_model.forward_one(img)

                # Compare distances to reference embeddings
                pos_distance = adapted_model.compute_distance(
                    embedding, pos_ref_embedding
                ).item()
                neg_distance = adapted_model.compute_distance(
                    embedding, neg_ref_embedding
                ).item()

                # Classify based on closer reference
                pred = 1 if pos_distance < neg_distance else 0
                predictions.append(pred)
                true_labels.append(1)  # This is a positive case

            for idx in tqdm(query_neg, desc="Evaluating negative cases"):
                img, _, _ = test_dataset[idx]
                img = img.unsqueeze(0).to(device)
                embedding = adapted_model.forward_one(img)

                # Compare distances to reference embeddings
                pos_distance = adapted_model.compute_distance(
                    embedding, pos_ref_embedding
                ).item()
                neg_distance = adapted_model.compute_distance(
                    embedding, neg_ref_embedding
                ).item()

                # Classify based on closer reference
                pred = 1 if pos_distance < neg_distance else 0
                predictions.append(pred)
                true_labels.append(0)  # This is a negative case

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)

        print(f"Disease: {disease_name}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")

        results[disease_name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "predictions": predictions,
            "true_labels": true_labels,
        }

    # Print overall results
    print("\n=== Overall Results ===")
    if results:
        avg_accuracy = np.mean([res["accuracy"] for res in results.values()])
        avg_precision = np.mean([res["precision"] for res in results.values()])
        avg_recall = np.mean([res["recall"] for res in results.values()])

        print(f"Average accuracy across diseases: {avg_accuracy:.4f}")
        print(f"Average precision across diseases: {avg_precision:.4f}")
        print(f"Average recall across diseases: {avg_recall:.4f}")
    else:
        print("No valid results to report")

    return results


def visualize_results(results, save_dir="results"):
    """
    Visualize prediction results

    Args:
        results: Dictionary with results by disease
        save_dir: Directory to save visualization
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create figure for overall results
    plt.figure(figsize=(10, 6))
    diseases = list(results.keys())
    accuracies = [results[d]["accuracy"] for d in diseases]
    precisions = [results[d]["precision"] for d in diseases]
    recalls = [results[d]["recall"] for d in diseases]

    x = np.arange(len(diseases))
    width = 0.25

    plt.bar(x - width, accuracies, width, label="Accuracy")
    plt.bar(x, precisions, width, label="Precision")
    plt.bar(x + width, recalls, width, label="Recall")

    plt.ylabel("Score")
    plt.title("Performance by Disease")
    plt.xticks(x, diseases, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/overall_performance.png")

    # Create confusion matrices for each disease
    for disease, res in results.items():
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(res["true_labels"], res["predictions"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix: {disease}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix_{disease.replace(' ', '_')}.png")
        plt.close()

    print(f"Visualizations saved to {save_dir}/")


if __name__ == "__main__":
    results = predict_with_meta_adaptation()
    visualize_results(results)
