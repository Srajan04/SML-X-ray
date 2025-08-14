import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import yaml
import torchvision.transforms as transforms
from pathlib import Path


def cache_preprocessed_images(config_path="config/config.yaml"):
    """
    Preprocess and cache all images to disk as tensors
    """
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Get paths
    dataset_path = config["data"]["dataset_path"]
    cache_dir = os.path.join(dataset_path, "preprocessed_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Define transform (without augmentation for caching)
    transform = transforms.Compose(
        [
            transforms.Resize(config["data"]["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Process CSVs
    for split in ["train", "valid", "test"]:
        print(f"Processing {split} split...")
        if split == "test" or split == "valid":
            csv_path = os.path.join(dataset_path, "valid_balanced.csv")
        else:
            csv_path = os.path.join(dataset_path, "train_balanced_reduced_40pct.csv")

        if not os.path.exists(csv_path):
            print(f"CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Create cache for this split
        split_cache_dir = os.path.join(cache_dir, split)
        os.makedirs(split_cache_dir, exist_ok=True)

        # Process each image in the CSV
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_path = row["Path"]

            # Build the full path
            if img_path.startswith("CheXpert"):
                full_path = os.path.join(dataset_path, img_path)
            else:
                full_path = os.path.join(dataset_path, "CheXpert-v1.0-small", img_path)

            # Create a unique ID for the cached file
            img_id = f"{idx}_{Path(img_path).stem}"
            cache_path = os.path.join(split_cache_dir, f"{img_id}.pt")

            # Skip if already processed
            if os.path.exists(cache_path):
                continue

            try:
                # Load and transform the image
                img = Image.open(full_path).convert("RGB")
                img_tensor = transform(img)

                # Save the tensor to disk
                torch.save(img_tensor, cache_path)
            except Exception as e:
                print(f"Error processing {full_path}: {e}")

        # Create a mapping CSV
        cache_df = df.copy()
        cache_df["cached_path"] = [
            os.path.join(split_cache_dir, f"{idx}_{Path(row['Path']).stem}.pt")
            for idx, row in df.iterrows()
        ]
        cache_df.to_csv(os.path.join(cache_dir, f"{split}_cache.csv"), index=False)

    print("Preprocessing complete. Updating config...")

    # Update config with cache paths
    config["data"]["use_cache"] = True
    config["data"]["cache_dir"] = cache_dir

    # Save updated config
    with open(f"{config_path}.cached", "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Updated config saved to {config_path}.cached")


if __name__ == "__main__":
    cache_preprocessed_images()
