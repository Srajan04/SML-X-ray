import os
import pandas as pd
import yaml
import numpy as np
from sklearn.model_selection import train_test_split


def create_balanced_splits(config_path):
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    dataset_path = config["data"]["dataset_path"]
    print(f"Creating balanced splits for data in {dataset_path}")

    # Paths to CheXpert CSV files
    train_path = os.path.join(dataset_path, "train.csv")
    orig_val_path = os.path.join(dataset_path, "valid.csv")

    # Read the CSV files
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(orig_val_path)

    # Extract patient IDs if needed
    if "PatientID" not in train_df.columns:
        train_df["PatientID"] = train_df["Path"].apply(
            lambda x: x.split("/")[2] if len(x.split("/")) > 2 else "unknown"
        )
    if "PatientID" not in val_df.columns:
        val_df["PatientID"] = val_df["Path"].apply(
            lambda x: x.split("/")[2] if len(x.split("/")) > 2 else "unknown"
        )

    # Get unique patients
    train_patients = train_df["PatientID"].unique()

    # Create a new split - sample 15% of training patients for validation
    # This ensures no patient overlap between splits
    new_train_patients, new_val_patients = train_test_split(
        train_patients, test_size=0.15, random_state=42
    )

    # Create new dataframes
    new_train_df = train_df[train_df["PatientID"].isin(new_train_patients)]
    additional_val_df = train_df[train_df["PatientID"].isin(new_val_patients)]

    # Combine existing validation with additional validation samples
    new_val_df = pd.concat([val_df, additional_val_df], ignore_index=True)

    # Save new splits
    new_train_path = os.path.join(dataset_path, "train_balanced.csv")
    new_val_path = os.path.join(dataset_path, "valid_balanced.csv")

    new_train_df.to_csv(new_train_path, index=False)
    new_val_df.to_csv(new_val_path, index=False)

    print(f"Original train size: {len(train_df)}, Original val size: {len(val_df)}")
    print(f"New train size: {len(new_train_df)}, New val size: {len(new_val_df)}")

    return new_train_path, new_val_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    create_balanced_splits(args.config)
