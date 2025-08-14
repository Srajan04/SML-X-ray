import os
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
import sys
import matplotlib.pyplot as plt
import higher
import random
from tqdm import tqdm, trange

# Add the project root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.siamese import SiameseNetwork
from models.meta_learner import MAML, TaskGenerator
from data.dataloader import CheXpertDataset, get_data_loaders, CachedDataset


def train_meta_learner(config_path="config/config.yaml"):
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained Siamese model
    siamese_model = SiameseNetwork(
        backbone=config["siamese"]["backbone"],
        embedding_dim=config["siamese"]["embedding_dim"],
        pretrained=config["siamese"]["pretrained"],
    ).to(device)

    # Load pretrained weights if available
    checkpoint_path = os.path.join(
        config["training"]["save_dir"], "best_siamese_model.pth"
    )
    if os.path.exists(checkpoint_path):
        print(f"Loading pretrained Siamese model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        siamese_model.load_state_dict(checkpoint["model_state_dict"])

    # Ensure all parameters require gradients
    for param in siamese_model.parameters():
        param.requires_grad = True

    # Create MAML meta-learner
    meta_learner = MAML(
        model=siamese_model,
        inner_lr=config["meta"]["inner_lr"],
        first_order=(
            config["meta"]["algorithm"] == "fomaml"
        ),  # First-order approximation
    ).to(device)

    # Check if we should use cached datasets
    use_cache = config["data"].get("use_cache", False)
    cache_dir = config["data"].get("cache_dir", None)

    if use_cache and cache_dir and os.path.exists(cache_dir):
        # Use cached datasets
        print(f"Using cached datasets from {cache_dir}")
        train_csv = os.path.join(cache_dir, "train_cache.csv")
        val_csv = os.path.join(cache_dir, "valid_cache.csv")

        if os.path.exists(train_csv) and os.path.exists(val_csv):
            # Create datasets from cached tensors
            train_dataset = CachedDataset(train_csv)
            val_dataset = CachedDataset(val_csv)
        else:
            print("Cache CSVs not found. Falling back to regular loading.")
            # Use regular datasets with paths from config
            train_csv = os.path.join(
                config["data"]["dataset_path"], config["data"]["train_csv"]
            )
            val_csv = os.path.join(
                config["data"]["dataset_path"], config["data"]["valid_csv"]
            )

            # Define transformations
            transform = transforms.Compose(
                [
                    transforms.Resize(config["data"]["image_size"]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # Create datasets with use_cache=True to try to use cache if available
            train_dataset = CheXpertDataset(
                train_csv,
                config["data"]["dataset_path"],
                transform=transform,
                use_cache=True,
            )
            val_dataset = CheXpertDataset(
                val_csv,
                config["data"]["dataset_path"],
                transform=transform,
                use_cache=True,
            )
    else:
        # Use regular datasets with paths from config
        train_csv = os.path.join(
            config["data"]["dataset_path"], config["data"]["train_csv"]
        )
        val_csv = os.path.join(
            config["data"]["dataset_path"], config["data"]["valid_csv"]
        )

        # Define transformations
        transform = transforms.Compose(
            [
                transforms.Resize(config["data"]["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Create datasets
        train_dataset = CheXpertDataset(
            train_csv, config["data"]["dataset_path"], transform=transform
        )
        val_dataset = CheXpertDataset(
            val_csv, config["data"]["dataset_path"], transform=transform
        )

    # Create task generators
    train_task_generator = TaskGenerator(
        dataset=train_dataset,
        n_way=2,  # Binary classification
        k_shot=config["meta"]["shots"],
        query_size=config["meta"]["shots"] * 2,
    )

    val_task_generator = TaskGenerator(
        dataset=val_dataset,
        n_way=2,  # Binary classification
        k_shot=config["meta"]["shots"],
        query_size=config["meta"]["shots"] * 2,
    )

    # Define meta-optimizer
    meta_optimizer = optim.Adam(
        meta_learner.parameters(),
        lr=config["meta"]["meta_lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    # Meta-learning training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    # Create directory for saving checkpoints
    save_dir = os.path.join(os.getcwd(), config["training"]["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    # Set batch size for tasks - adjust based on your VRAM
    task_batch_size = 1  # Increase this if you have more VRAM

    # Meta-training process with tqdm progress bars
    for episode in trange(
        config["training"]["meta_episodes"], desc="Meta-training episodes"
    ):
        # Training phase
        meta_learner.train()
        episode_loss = 0.0
        tasks_processed = 0

        # Process tasks in batches
        for batch_idx in range(0, config["meta"]["tasks_per_batch"], task_batch_size):
            # Get actual batch size for this iteration
            curr_batch_size = min(
                task_batch_size, config["meta"]["tasks_per_batch"] - batch_idx
            )

            # Initialize batch data containers
            batch_support_losses = []
            batch_query_losses = []

            # Sample tasks for this batch
            valid_tasks = 0
            for task_idx in range(curr_batch_size):
                # Sample a task (support and query sets)
                support_pairs, query_pairs = train_task_generator.sample_task()

                # Skip if no support or query pairs
                if len(support_pairs) == 0:
                    continue

                # Check if there are query pairs, create from support if needed
                if len(query_pairs) == 0:
                    if len(support_pairs) >= 4:
                        # Shuffle support pairs and use some as query pairs
                        random.shuffle(support_pairs)
                        mid = len(support_pairs) // 2
                        query_pairs = support_pairs[mid:]
                        support_pairs = support_pairs[:mid]
                    else:
                        # Not enough pairs to split, duplicate
                        query_pairs = support_pairs.copy()

                # Prepare data
                support_x1 = torch.stack([pair[0] for pair in support_pairs]).to(device)
                support_x2 = torch.stack([pair[1] for pair in support_pairs]).to(device)
                support_y = torch.stack([pair[2] for pair in support_pairs]).to(device)
                query_x1 = torch.stack([pair[0] for pair in query_pairs]).to(device)
                query_x2 = torch.stack([pair[1] for pair in query_pairs]).to(device)
                query_y = torch.stack([pair[2] for pair in query_pairs]).to(device)

                # Clean up memory by explicitly deleting unused variables
                if task_idx > 0:
                    # Clear references to previous tensors to help garbage collection
                    del support_pairs, query_pairs
                    torch.cuda.empty_cache()  # Free up GPU memory

                # Run meta-learning step with higher library
                with higher.innerloop_ctx(
                    meta_learner.model,
                    torch.optim.SGD(
                        meta_learner.model.parameters(), lr=config["meta"]["inner_lr"]
                    ),
                    track_higher_grads=not meta_learner.first_order,
                    copy_initial_weights=True,
                ) as (fmodel, diffopt):
                    # Inner loop adaptation on support set
                    for _ in range(config["meta"]["adapt_steps"]):
                        # Forward pass on support set
                        support_emb1, support_emb2 = fmodel(support_x1, support_x2)
                        support_distance = fmodel.compute_distance(
                            support_emb1, support_emb2
                        )

                        # Compute loss
                        support_loss = F.binary_cross_entropy_with_logits(
                            1.0 - support_distance, support_y.float()
                        )

                        # Update model parameters
                        diffopt.step(support_loss)

                    # Evaluate on query set with adapted model
                    query_emb1, query_emb2 = fmodel(query_x1, query_x2)
                    query_distance = fmodel.compute_distance(query_emb1, query_emb2)

                    # Compute query loss
                    query_loss = F.binary_cross_entropy_with_logits(
                        1.0 - query_distance, query_y.float()
                    )
                    batch_query_losses.append(query_loss)

                valid_tasks += 1

            # If we have at least one valid task
            if valid_tasks > 0:
                # Combine losses from all tasks in batch
                batch_loss = torch.mean(torch.stack(batch_query_losses))

                # Optimize meta-model
                meta_optimizer.zero_grad()
                batch_loss.backward()
                meta_optimizer.step()

                # Update metrics
                episode_loss += batch_loss.item()
                tasks_processed += valid_tasks

                # Clear memory
                del batch_query_losses
                torch.cuda.empty_cache()

        # Average loss over tasks processed
        if tasks_processed > 0:
            episode_loss /= tasks_processed
        train_losses.append(episode_loss)

        # Print progress
        if (episode + 1) % config["training"]["log_interval"] == 0:
            print(
                f"Episode {episode + 1}/{config['training']['meta_episodes']}, Loss: {episode_loss:.4f}, Tasks: {tasks_processed}"
            )

        # Validation phase (every 5 episodes)
        if (episode + 1) % 10 == 0:
            meta_learner.eval()
            val_episode_loss = 0.0
            val_tasks_processed = 0

            # Validate in batches too
            for batch_idx in range(
                0, config["meta"]["tasks_per_batch"], task_batch_size
            ):
                curr_batch_size = min(
                    task_batch_size, config["meta"]["tasks_per_batch"] - batch_idx
                )

                for val_task_idx in range(curr_batch_size):
                    # Sample a validation task
                    support_pairs, query_pairs = val_task_generator.sample_task()

                    # Skip if no support pairs
                    if len(support_pairs) == 0:
                        continue

                    # Prepare data
                    support_x1 = torch.stack([pair[0] for pair in support_pairs]).to(
                        device
                    )
                    support_x2 = torch.stack([pair[1] for pair in support_pairs]).to(
                        device
                    )
                    support_y = torch.stack([pair[2] for pair in support_pairs]).to(
                        device
                    )

                    # Check if there are query pairs
                    if len(query_pairs) == 0:
                        query_pairs = support_pairs.copy()

                    query_x1 = torch.stack([pair[0] for pair in query_pairs]).to(device)
                    query_x2 = torch.stack([pair[1] for pair in query_pairs]).to(device)
                    query_y = torch.stack([pair[2] for pair in query_pairs]).to(device)

                    # Create a separate model copy for validation
                    model_copy = SiameseNetwork(
                        backbone=config["siamese"]["backbone"],
                        embedding_dim=config["siamese"]["embedding_dim"],
                        pretrained=False,
                    ).to(device)
                    model_copy.load_state_dict(meta_learner.model.state_dict())

                    # Enable gradients for adaptation phase
                    model_copy.train()
                    for param in model_copy.parameters():
                        param.requires_grad = True

                    # Manually adapt the model using SGD
                    optimizer = torch.optim.SGD(
                        model_copy.parameters(), lr=config["meta"]["inner_lr"]
                    )

                    # Adaptation steps
                    for _ in range(config["meta"]["adapt_steps"]):
                        # Forward pass
                        emb1, emb2 = model_copy(support_x1, support_x2)
                        distance = model_copy.compute_distance(emb1, emb2)
                        loss = F.binary_cross_entropy_with_logits(
                            1.0 - distance, support_y.float()
                        )

                        # Update parameters
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Switch to eval mode for final evaluation
                    model_copy.eval()

                    # Evaluate adapted model without tracking gradients
                    with torch.no_grad():
                        emb1, emb2 = model_copy(query_x1, query_x2)
                        distance = model_copy.compute_distance(emb1, emb2)
                        val_task_loss = F.binary_cross_entropy_with_logits(
                            1.0 - distance, query_y.float()
                        )

                    val_episode_loss += val_task_loss.item()
                    val_tasks_processed += 1

                    # Clean up VRAM
                    del (
                        model_copy,
                        support_x1,
                        support_x2,
                        support_y,
                        query_x1,
                        query_x2,
                        query_y,
                    )
                    del emb1, emb2, distance
                    torch.cuda.empty_cache()

            # Calculate average validation loss
            if val_tasks_processed > 0:
                val_episode_loss /= val_tasks_processed
                val_losses.append(val_episode_loss)
                print(f"Validation Loss: {val_episode_loss:.4f}")

                # Save best model
                if val_episode_loss < best_val_loss:
                    best_val_loss = val_episode_loss
                    torch.save(
                        {
                            "episode": episode,
                            "model_state_dict": meta_learner.model.state_dict(),
                            "optimizer_state_dict": meta_optimizer.state_dict(),
                            "loss": val_episode_loss,
                        },
                        os.path.join(save_dir, "best_meta_model.pth"),
                    )
                    print(
                        f"Saved best model with validation loss: {val_episode_loss:.4f}"
                    )

        # Save checkpoint every 100 episodes
        if (episode + 1) % 100 == 0:
            torch.save(
                {
                    "episode": episode,
                    "model_state_dict": meta_learner.model.state_dict(),
                    "optimizer_state_dict": meta_optimizer.state_dict(),
                    "loss": episode_loss,
                },
                os.path.join(save_dir, f"meta_episode_{episode + 1}.pth"),
            )

    # Plot and save training curve
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(train_losses)), train_losses, label="Training Loss")
    plt.plot(np.arange(len(val_losses)) * 5, val_losses, label="Validation Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Meta-Learning Training and Validation Losses")
    plt.savefig(os.path.join(save_dir, "meta_training_curve.png"))
    plt.close()

    print("Meta-training completed!")
    return meta_learner
