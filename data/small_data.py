import pandas as pd
import os
import shutil
from pathlib import Path


def create_dataset_subset(
    original_csv_path,
    output_csv_path,
    output_images_dir,
    subset_size=1000,
    random_seed=42,
):
    """Create a smaller subset of the CheXpert dataset"""

    # Read the original CSV file
    df = pd.read_csv(original_csv_path)

    # Sample a subset randomly
    subset_df = df.sample(n=min(subset_size, len(df)), random_state=random_seed)

    # Create output directory
    os.makedirs(output_images_dir, exist_ok=True)

    # Copy images to subset directory and update paths
    image_paths = []
    base_dir = os.path.dirname(original_csv_path)

    for idx, row in subset_df.iterrows():
        original_path = os.path.join(base_dir, row["Path"])

        if os.path.exists(original_path):
            # Create relative path structure in the destination
            rel_path = os.path.relpath(original_path, base_dir)
            dest_path = os.path.join(output_images_dir, rel_path)

            # Create directories if needed
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Copy the image file
            shutil.copy2(original_path, dest_path)

            # Store the new path
            image_paths.append(rel_path)
        else:
            image_paths.append(None)

    # Update paths in the dataframe
    subset_df["Path"] = image_paths

    # Remove rows with missing images
    subset_df = subset_df[subset_df["Path"].notna()]

    # Save the subset CSV
    subset_df.to_csv(output_csv_path, index=False)

    print(f"Created dataset subset with {len(subset_df)} samples")
    return subset_df


if __name__ == "__main__":
    # Usage example - modify paths for your setup
    dataset_path = "/home/srajan/Data/Datasets_ATML_Projects"

    # Create subsets for train, validation, and test
    create_dataset_subset(
        original_csv_path=os.path.join(dataset_path, "train_balanced.csv"),
        output_csv_path=os.path.join(dataset_path, "train_small.csv"),
        output_images_dir=os.path.join(dataset_path, "CheXpert-small-subset"),
        subset_size=190359,
    )

    create_dataset_subset(
        original_csv_path=os.path.join(dataset_path, "valid_balanced.csv"),
        output_csv_path=os.path.join(dataset_path, "valid_small.csv"),
        output_images_dir=os.path.join(dataset_path, "CheXpert-small-subset"),
        subset_size=33289,
    )

    create_dataset_subset(
        original_csv_path=os.path.join(dataset_path, "valid_balanced.csv"),
        output_csv_path=os.path.join(dataset_path, "test_small.csv"),
        output_images_dir=os.path.join(dataset_path, "CheXpert-small-subset"),
        subset_size=33289,
    )
