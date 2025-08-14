import os
import torch
import yaml
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from models.siamese import SiameseNetwork
from data.dataloader import CheXpertDataset, CachedDataset


# Function to load model
def load_model(model_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    return siamese_model, device


# Function to preprocess image
def preprocess_image(image, config):
    transform = transforms.Compose(
        [
            transforms.Resize(
                (config["data"]["image_size"][0], config["data"]["image_size"][1])
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)  # Add batch dimension


# Function to adapt model with support images
def adapt_model(model, support_images, config, device):
    # Create a copy of the model for adaptation
    adapted_model = SiameseNetwork(
        backbone=config["siamese"]["backbone"],
        embedding_dim=config["siamese"]["embedding_dim"],
        pretrained=False,
    ).to(device)
    adapted_model.load_state_dict(model.state_dict())
    adapted_model.train()

    # Skip adaptation if less than 2 support images
    if len(support_images) < 2:
        return adapted_model

    # Create pairs from support images (assuming all are similar)
    support_tensors = [
        preprocess_image(img, config).to(device) for img in support_images
    ]

    # Create all possible pairs (assuming all support images are from same disease)
    support_pairs = []
    for i in range(len(support_tensors)):
        for j in range(i + 1, len(support_tensors)):
            support_pairs.append(
                (support_tensors[i], support_tensors[j], torch.tensor(1.0).to(device))
            )

    if len(support_pairs) == 0:
        return adapted_model

    # Prepare data for adaptation
    support_x1 = torch.cat([p[0] for p in support_pairs])
    support_x2 = torch.cat([p[1] for p in support_pairs])
    support_y = torch.stack([p[2] for p in support_pairs])

    # Adaptation loop with improved optimization
    adapt_steps = config["meta"].get("adapt_steps", 5) * 2  # Double the steps
    optimizer = torch.optim.SGD(
        adapted_model.parameters(),
        lr=config["meta"]["inner_lr"],
        momentum=0.9,  # Add momentum for better convergence
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=adapt_steps // 2, gamma=0.5
    )

    for step in range(adapt_steps):
        emb1, emb2 = adapted_model(support_x1, support_x2)
        distance = adapted_model.compute_distance(emb1, emb2)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            1.0 - distance, support_y.float()
        )
        optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    adapted_model.eval()
    return adapted_model


def ensemble_direct_predictions(
    model,
    image,
    config,
    device,
    disease_references,
    prediction_mode,
    num_samples=5,
    temperature=0.5,
):
    """
    Generate predictions for multiple augmented versions of the image, then average results.

    Args:
        model: The base prediction model.
        image: The PIL.Image object.
        config: Configuration dictionary.
        device: Torch device.
        disease_references: Reference embeddings for each disease.
        prediction_mode: String indicator, e.g. "Direct Prediction".
        num_samples: Number of augmented crops to generate.

    Returns:
        Averaged prediction results dictionary.
    """
    from torchvision import transforms

    # Define an augmentation transform for ensemble predictions.
    # Adjust the transform as needed (here we use RandomResizedCrop).
    augmentation_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=config["data"]["image_size"], scale=(0.8, 1.0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    all_results = []
    for _ in range(num_samples):
        # Apply augmentation on the original image
        augmented_image = augmentation_transform(image)
        # Add batch dimension
        image_tensor = augmented_image.unsqueeze(0).to(device)
        # Get predictions using the existing function
        result = predict_diseases(
            model,
            image_tensor,
            disease_references,
            device,
            prediction_mode,
            temperature=temperature,
        )
        all_results.append(result)

    # Average results for each disease across all augmentations
    averaged_results = {}
    for disease in disease_references.keys():
        probs = []
        pos_sims = []
        neg_sims = []
        confs = []
        for res in all_results:
            if disease in res:
                probs.append(res[disease]["probability"])
                pos_sims.append(res[disease]["positive_similarity"])
                neg_sims.append(res[disease]["negative_similarity"])
                confs.append(res[disease]["confidence"])
        if probs:
            averaged_results[disease] = {
                "probability": np.mean(probs),
                "positive_similarity": np.mean(pos_sims),
                "negative_similarity": np.mean(neg_sims),
                "confidence": np.mean(confs),
            }
    return averaged_results


# Function to predict diseases
def predict_diseases(
    model, image_tensor, disease_references, device, prediction_mode, temperature=0.5
):
    """
    Predict diseases by comparing with reference embeddings
    """
    with torch.no_grad():
        # Get embedding for the query image
        query_embedding = model.forward_one(image_tensor)

        # Compare with reference embeddings for each disease
        results = {}
        for disease_name, (pos_emb, neg_emb) in disease_references.items():
            # Calculate distances
            pos_distance = model.compute_distance(query_embedding, pos_emb).item()
            neg_distance = model.compute_distance(query_embedding, neg_emb).item()

            # Apply temperature scaling for better calibration
            # temperature = (
            #     0.2 if prediction_mode == "Direct Prediction" else 0.5
            # )  # Lower temperature = sharper distinctions
            pos_similarity = (1.0 - pos_distance) / temperature
            neg_similarity = (1.0 - neg_distance) / temperature

            # Calculate probability (softmax of similarities)
            total = np.exp(pos_similarity) + np.exp(neg_similarity)
            probability = np.exp(pos_similarity) / total

            # Add confidence metric based on distance margin
            margin = neg_distance - pos_distance
            confidence = np.tanh(margin * 2)  # Scale and bound the confidence

            results[disease_name] = {
                "probability": probability,
                "positive_similarity": pos_similarity,
                "negative_similarity": neg_similarity,
                "confidence": confidence,
            }

        return results


# Function to create reference embeddings
def create_reference_embeddings(config, model, device):
    """
    Create reference embeddings for each disease from the test dataset
    """
    # Load test dataset
    use_cache = config["data"].get("use_cache", False)
    cache_dir = config["data"].get("cache_dir", None)

    if (
        use_cache
        and cache_dir
        and os.path.exists(os.path.join(cache_dir, "test_cache.csv"))
    ):
        test_dataset = CachedDataset(os.path.join(cache_dir, "test_cache.csv"))
    else:
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
            config["data"]["dataset_path"], config["data"]["test_csv"]
        )
        test_dataset = CheXpertDataset(
            test_csv, config["data"]["dataset_path"], transform=transform
        )

    # Get disease names
    disease_names = test_dataset.target_diseases

    # Create reference embeddings for each disease
    reference_embeddings = {}

    for disease_idx, disease_name in enumerate(disease_names):
        # Find positive and negative cases
        pos_indices = []
        neg_indices = []

        # Increase search range for more robust embedding creation
        for i in range(
            min(500, len(test_dataset))
        ):  # Increased from 200 to 500 for better coverage
            _, label, _ = test_dataset[i]
            if label[disease_idx] == 1:
                pos_indices.append(i)
            elif label[disease_idx] == 0:
                neg_indices.append(i)

            # Stop if we have enough samples
            if (
                len(pos_indices) >= 10 and len(neg_indices) >= 10
            ):  # Increased from 5 to 10
                break

        # Skip diseases with too few samples
        if len(pos_indices) < 3 or len(neg_indices) < 3:
            continue

        # Average positive embeddings - use more samples for better representation
        pos_ref_embedding = torch.zeros(1, config["siamese"]["embedding_dim"]).to(
            device
        )
        samples_to_use = min(10, len(pos_indices))  # Increased from 5 to 10
        for idx in pos_indices[:samples_to_use]:
            img, _, _ = test_dataset[idx]
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.forward_one(img)
                pos_ref_embedding += embedding
        pos_ref_embedding /= samples_to_use

        # Average negative embeddings - use more samples for better representation
        neg_ref_embedding = torch.zeros(1, config["siamese"]["embedding_dim"]).to(
            device
        )
        samples_to_use = min(10, len(neg_indices))  # Increased from 5 to 10
        for idx in neg_indices[:samples_to_use]:
            img, _, _ = test_dataset[idx]
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.forward_one(img)
                neg_ref_embedding += embedding
        neg_ref_embedding /= samples_to_use

        # Store reference embeddings
        reference_embeddings[disease_name] = (pos_ref_embedding, neg_ref_embedding)

    return reference_embeddings


# Main Streamlit app
def main():
    st.set_page_config(page_title="X-ray Disease Predictor", layout="wide")

    st.title("X-ray Disease Prediction")
    st.write("""
    Upload an X-ray image to predict potential diseases.
    Optionally upload support images for model adaptation.
    """)

    # Load configuration
    config_path = "config/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Sidebar for model selection
    st.sidebar.title("Model Settings")
    model_type = st.sidebar.radio(
        "Select Model Type", ("Standard Siamese", "Meta-Learning (with adaptation)")
    )

    prediction_mode = st.sidebar.radio(
        "Prediction Mode", ("Direct Prediction", "Adaptation with Reference Library")
    )

    # Add temperature slider for both prediction modes
    st.sidebar.subheader("Prediction Settings")
    temperature = st.sidebar.slider(
        "Temperature",
        0.1,
        1.0,
        0.5,
        0.05,
        help="Lower values make predictions more decisive, higher values make them smoother",
    )

    # Add contrast factor slider for direct prediction
    dp_contrast = 5.0
    if prediction_mode == "Direct Prediction":
        dp_contrast = st.sidebar.slider(
            "Contrast Factor",
            1.0,
            10.0,
            5.0,
            0.5,
            help="Higher values increase contrast between probabilities",
        )

    # Add adapt steps slider for adaptation mode
    adapt_steps = config["meta"]["adapt_steps"]
    if prediction_mode == "Adaptation with Reference Library":
        adapt_steps = st.sidebar.slider(
            "Adaptation Steps",
            1,
            20,
            config["meta"]["adapt_steps"],
            1,
            help="Number of gradient updates during adaptation",
        )

    # Load appropriate model
    if model_type == "Standard Siamese":
        model_path = os.path.join(
            config["training"]["save_dir"], "best_siamese_model.pth"
        )
        model_name = "Standard Siamese"
    else:
        model_path = os.path.join(config["training"]["save_dir"], "best_meta_model.pth")
        model_name = "Meta-Learning"

    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return

    # Load the model
    with st.spinner("Loading model..."):
        model, device = load_model(model_path, config)
        st.sidebar.success(f"Loaded {model_name} model")

    # Create reference embeddings
    with st.spinner("Creating reference embeddings..."):
        reference_embeddings = create_reference_embeddings(config, model, device)
        st.sidebar.success(
            f"Created reference embeddings for {len(reference_embeddings)} diseases"
        )

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.header("Input")
        uploaded_file = st.file_uploader(
            "Upload an X-ray image", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded X-ray", width=300)

            # Preprocess the image
            image_tensor = preprocess_image(image, config).to(device)

            # Handle the two prediction modes
            if prediction_mode == "Direct Prediction":
                # Use the base model without adaptation for all diseases
                prediction_model = model
            else:
                # Reference Library Mode - use pre-loaded reference images for each disease
                st.subheader("Disease-Specific Adaptation")
                st.write("Select diseases to consider for targeted adaptation")

                # Get available diseases
                available_diseases = list(reference_embeddings.keys())
                selected_diseases = st.multiselect(
                    "Select specific diseases to check (or leave empty to check all)",
                    available_diseases,
                    default=[],
                )

                if not selected_diseases:  # If none selected, use all
                    selected_diseases = available_diseases

                # Create container for results from each adapted model
                all_disease_results = {}

            # Prediction button
            if st.button("Analyze X-ray"):
                if prediction_mode == "Direct Prediction":
                    # Standard prediction using base model
                    with st.spinner("Analyzing X-ray..."):
                        # results = predict_diseases(
                        #     prediction_model,
                        #     image_tensor,
                        #     reference_embeddings,
                        #     device,
                        #     prediction_mode="Direct Prediction",
                        #     temperature=temperature
                        # )
                        # normalized_results = normalize_multi_model_results(
                        #     results, contrast_factor=dp_contrast
                        # )
                        # display_results(normalized_results, col2)
                        ensemble_results = ensemble_direct_predictions(
                            prediction_model,
                            image,
                            config,
                            device,
                            reference_embeddings,
                            prediction_mode="Direct Prediction",
                            num_samples=15,
                            temperature=temperature,
                        )
                        normalized_results = normalize_multi_model_results(
                            ensemble_results, contrast_factor=dp_contrast
                        )
                        display_results(normalized_results, col2)

                else:  # Reference Library Mode
                    # For each disease, adapt the model and predict
                    all_results = {}
                    progress_bar = st.progress(0)

                    # Process diseases one at a time with memory cleanup
                    for i, disease in enumerate(selected_diseases):
                        # Update progress
                        progress_bar.progress((i + 1) / len(selected_diseases))
                        st.write(f"Adapting for {disease}...")

                        # Clear CUDA cache before each adaptation
                        torch.cuda.empty_cache()
                        import gc

                        gc.collect()

                        # Use reference library for adaptation
                        adapted_model = adapt_model_for_disease(
                            model, disease, config, device, adapt_steps=adapt_steps
                        )

                        # Use ensemble prediction combining base and adapted models
                        disease_result = ensemble_prediction(
                            model,  # Base model
                            adapted_model,  # Adapted model
                            image_tensor,
                            disease,
                            reference_embeddings[disease],
                            device,
                            alpha=0.7,  # Weight for adapted model
                        )

                        all_results[disease] = disease_result

                        # Delete the adapted model to free memory
                        del adapted_model
                        torch.cuda.empty_cache()

                    # Combine and normalize results with improved normalization
                    final_results = normalize_multi_model_results(all_results)
                    display_results(final_results, col2)

        else:
            with col2:
                st.header("Prediction Results")
                st.write("Please upload an X-ray image to get predictions.")


def adapt_model_for_disease(base_model, disease, config, device, adapt_steps=None):
    """
    Adapt the model using reference images for a specific disease
    from the CheXpert dataset
    """
    # Load the validation dataset to use as a reference library
    use_cache = config["data"].get("use_cache", False)
    cache_dir = config["data"].get("cache_dir", None)

    if (
        use_cache
        and cache_dir
        and os.path.exists(os.path.join(cache_dir, "valid_cache.csv"))
    ):
        dataset = CachedDataset(os.path.join(cache_dir, "valid_cache.csv"))
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(config["data"]["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_csv = os.path.join(
            config["data"]["dataset_path"], config["data"]["valid_csv"]
        )
        dataset = CheXpertDataset(
            val_csv, config["data"]["dataset_path"], transform=transform
        )

    # Create a copy of the base model
    adapted_model = SiameseNetwork(
        backbone=config["siamese"]["backbone"],
        embedding_dim=config["siamese"]["embedding_dim"],
        pretrained=False,
    ).to(device)
    adapted_model.load_state_dict(base_model.state_dict())
    adapted_model.train()

    # Find the disease index
    disease_idx = dataset.target_diseases.index(disease)

    # Collect positive and negative examples
    pos_indices = []
    neg_indices = []

    # Increase the search range to find better examples
    max_samples_to_check = min(500, len(dataset))

    for i in range(max_samples_to_check):
        _, label, _ = dataset[i]
        if label[disease_idx] == 1:
            pos_indices.append(i)
        elif label[disease_idx] == 0:
            neg_indices.append(i)

        # Once we have enough samples, stop searching
        if len(pos_indices) >= 15 and len(neg_indices) >= 15:  # Increased from 10 to 15
            break

    # If we don't have enough samples, return the base model
    if len(pos_indices) < 3 or len(neg_indices) < 3:
        print(f"Not enough samples for {disease} adaptation")
        return adapted_model

    # Increase sample count for adaptation
    pos_indices = pos_indices[: min(10, len(pos_indices))]  # Increased from 5 to 10
    neg_indices = neg_indices[: min(10, len(neg_indices))]  # Increased from 5 to 10

    # Create support pairs
    support_pairs = []

    # Similar pairs (pos-pos)
    for i in range(len(pos_indices)):
        for j in range(i + 1, len(pos_indices)):
            img1, _, _ = dataset[pos_indices[i]]
            img2, _, _ = dataset[pos_indices[j]]
            support_pairs.append((img1, img2, torch.tensor(1.0)))

    # Similar pairs (neg-neg)
    for i in range(len(neg_indices)):
        for j in range(i + 1, len(neg_indices)):
            img1, _, _ = dataset[neg_indices[i]]
            img2, _, _ = dataset[neg_indices[j]]
            support_pairs.append((img1, img2, torch.tensor(1.0)))

    # Dissimilar pairs (pos-neg)
    for i in range(min(len(pos_indices), len(neg_indices))):
        img1, _, _ = dataset[pos_indices[i]]
        img2, _, _ = dataset[neg_indices[i]]
        support_pairs.append((img1, img2, torch.tensor(0.0)))

    # Skip if not enough pairs
    if len(support_pairs) < 3:
        print(f"Not enough pairs for {disease} adaptation")
        return adapted_model

    # Prepare data for adaptation
    support_x1 = torch.stack([p[0] for p in support_pairs]).to(device)
    support_x2 = torch.stack([p[1] for p in support_pairs]).to(device)
    support_y = torch.stack([p[2] for p in support_pairs]).to(device)

    # Use provided adapt_steps if specified, otherwise use from config
    if adapt_steps is None:
        adapt_steps = config["meta"]["adapt_steps"]

    # Adaptation loop with improved optimization
    optimizer = torch.optim.SGD(
        adapted_model.parameters(), lr=config["meta"]["inner_lr"], momentum=0.9
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=adapt_steps // 2, gamma=0.5
    )

    batch_size = 4
    for step in range(adapt_steps):
        for b in range(0, len(support_x1), batch_size):
            batch_x1 = support_x1[b : b + batch_size]
            batch_x2 = support_x2[b : b + batch_size]
            batch_y = support_y[b : b + batch_size]

            emb1, emb2 = adapted_model(batch_x1, batch_x2)
            distance = adapted_model.compute_distance(emb1, emb2)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                1.0 - distance, batch_y.float()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    adapted_model.eval()
    return adapted_model


def predict_specific_disease(
    model, image_tensor, disease, reference_embeddings, device
):
    """
    Make prediction for a specific disease with an adapted model
    """
    with torch.no_grad():
        query_embedding = model.forward_one(image_tensor)
        pos_emb, neg_emb = reference_embeddings

        # Calculate distances
        pos_distance = model.compute_distance(query_embedding, pos_emb).item()
        neg_distance = model.compute_distance(query_embedding, neg_emb).item()

        # Apply temperature scaling for better calibration
        temperature = 0.5  # Lower temperature = sharper distinctions
        pos_similarity = (1.0 - pos_distance) / temperature
        neg_similarity = (1.0 - neg_distance) / temperature

        # Calculate probability with modified softmax
        total = np.exp(pos_similarity) + np.exp(neg_similarity)
        probability = np.exp(pos_similarity) / total

        # Add confidence metric based on distance margin
        margin = neg_distance - pos_distance
        confidence = np.tanh(margin * 2)  # Scale and bound the confidence

        return {
            "probability": probability,
            "positive_similarity": pos_similarity,
            "negative_similarity": neg_similarity,
            "confidence": confidence,
        }


def ensemble_prediction(
    base_model,
    adapted_model,
    image_tensor,
    disease,
    reference_embeddings,
    device,
    alpha=0.7,
):
    """
    Ensemble prediction combining base and adapted model results
    """
    # Get predictions from both models
    adapted_result = predict_specific_disease(
        adapted_model, image_tensor, disease, reference_embeddings, device
    )
    base_result = predict_specific_disease(
        base_model, image_tensor, disease, reference_embeddings, device
    )

    # Weighted ensemble
    ensemble_prob = (alpha * adapted_result["probability"]) + (
        (1 - alpha) * base_result["probability"]
    )

    # Return combined result
    return {
        "probability": ensemble_prob,
        "positive_similarity": adapted_result[
            "positive_similarity"
        ],  # Keep adapted similarities
        "negative_similarity": adapted_result["negative_similarity"],
        "confidence": adapted_result.get("confidence", 0.5),
    }


def normalize_multi_model_results(all_results, contrast_factor=None):
    """
    Normalize probabilities across multiple disease-specific adapted models
    with improved confidence weighting
    """
    # Extract raw probabilities and confidence scores
    raw_probs = {
        disease: results["probability"] for disease, results in all_results.items()
    }
    confidences = {
        disease: results.get("confidence", 0.5)
        for disease, results in all_results.items()
    }

    # Apply weighted softmax based on confidence
    values = np.array(list(raw_probs.values()))
    confidence_weights = np.array(list(confidences.values()))

    # Use provided contrast factor or default dynamic value
    if contrast_factor is None:
        contrast_factor = 8 if max(values) > 0.9 else 5

    # Apply softmax with contrast and confidence weighting
    exp_values = np.exp(values * contrast_factor * (0.5 + confidence_weights / 2))
    softmax_values = exp_values / np.sum(exp_values)

    # Create normalized results
    normalized_results = {}
    for i, disease in enumerate(raw_probs.keys()):
        normalized_results[disease] = {
            "probability": softmax_values[i],
            "original_probability": all_results[disease]["probability"],
            "positive_similarity": all_results[disease]["positive_similarity"],
            "negative_similarity": all_results[disease]["negative_similarity"],
            "confidence": confidences.get(disease, 0.5),
        }

    return normalized_results


def display_results(results, column):
    """
    Display prediction results in the provided streamlit column
    """
    with column:
        st.header("Prediction Results")

        # Sort results by probability
        sorted_results = sorted(
            results.items(), key=lambda x: x[1]["probability"], reverse=True
        )

        # Display top predictions
        for disease, metrics in sorted_results:
            prob = metrics["probability"] * 100
            # Color code based on probability
            if prob > 70:
                color = "🔴"  # Red for high probability
            elif prob > 30:
                color = "🟠"  # Orange for medium
            else:
                color = "🟢"  # Green for low

            # Add confidence indicator if available
            confidence_str = ""
            if "confidence" in metrics:
                conf = metrics["confidence"]
                confidence_str = f" (Confidence: {conf:.2f})"

            st.write(f"{color} **{disease}**: {prob:.1f}%{confidence_str}")

            # Create a progress bar
            st.progress(metrics["probability"])

        # Plot probabilities as a bar chart
        st.subheader("Probability Distribution")
        fig, ax = plt.subplots(figsize=(10, len(sorted_results) * 0.4 + 1))

        diseases = [d for d, _ in sorted_results]
        probabilities = [m["probability"] * 100 for _, m in sorted_results]

        # Use a colormap based on probability
        colors = [
            "green" if p < 30 else "orange" if p < 70 else "red" for p in probabilities
        ]

        # Horizontal bar chart
        y_pos = np.arange(len(diseases))
        bars = ax.barh(y_pos, probabilities, align="center", color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(diseases)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel("Probability (%)")
        ax.set_title("Disease Probability")

        # Add probability values at the end of each bar
        for i, v in enumerate(probabilities):
            ax.text(v + 1, i, f"{v:.1f}%", va="center")

        st.pyplot(fig)

        # Show detailed analysis for top disease
        if len(sorted_results) > 0:
            top_disease, top_metrics = sorted_results[0]
            st.subheader(f"Detailed Analysis: {top_disease}")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Similarity Metrics:**")
                st.write(
                    f"- Positive similarity: {top_metrics['positive_similarity']:.4f}"
                )
                st.write(
                    f"- Negative similarity: {top_metrics['negative_similarity']:.4f}"
                )

            with col2:
                if "confidence" in top_metrics:
                    st.write(f"**Confidence Score:** {top_metrics['confidence']:.4f}")
                if "original_probability" in top_metrics:
                    st.write(
                        f"**Original probability:** {top_metrics['original_probability']:.4f}"
                    )


if __name__ == "__main__":
    main()
