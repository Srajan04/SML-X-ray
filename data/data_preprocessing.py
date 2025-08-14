import os
import pandas as pd
import numpy as np
import argparse
import yaml
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import shutil


def process_dataset(config_path):
    """
    Preprocess the CheXpert dataset:
    - Generate train/val/test splits based on patient ID
    - Handle uncertain labels
    - Create a balanced dataset for Siamese training
    """
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    dataset_path = config["data"]["dataset_path"]
    print(f"Processing data from {dataset_path}")

    # Paths to CheXpert CSV files
    train_path = os.path.join(dataset_path, "train_balanced_reduced_40pct.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"CheXpert data not found at {train_path}")

    # Read the main CSV file
    print("Reading CheXpert data...")
    df = pd.read_csv(train_path)

    # Extract patient IDs
    df["PatientID"] = df["Path"].apply(lambda x: x.split("/")[2])

    # Get unique patients
    patients = df["PatientID"].unique()
    print(f"Total patients: {len(patients)}")

    # Create patient splits (70% train, 15% val, 15% test)
    np.random.seed(42)
    np.random.shuffle(patients)

    train_split = int(0.7 * len(patients))
    val_split = int(0.85 * len(patients))

    train_patients = patients[:train_split]
    val_patients = patients[train_split:val_split]
    test_patients = patients[val_split:]

    print(f"Train patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")

    # Split data by patient
    train_df = df[df["PatientID"].isin(train_patients)]
    val_df = df[df["PatientID"].isin(val_patients)]
    test_df = df[df["PatientID"].isin(test_patients)]

    # Process uncertain labels (-1) by converting to positive (1)
    # This is a common approach in CheXpert
    disease_columns = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    for col in disease_columns:
        if col in train_df.columns:
            # Fill NaN with 0 (negative)
            train_df[col] = train_df[col].fillna(0)
            val_df[col] = val_df[col].fillna(0)
            test_df[col] = test_df[col].fillna(0)

            # Convert -1 (uncertain) to 1 (positive) - U-Ones approach
            train_df[col] = train_df[col].replace(-1, 1)
            val_df[col] = val_df[col].replace(-1, 1)
            test_df[col] = test_df[col].replace(-1, 1)

    # Save preprocessed CSVs
    train_df.to_csv(os.path.join(dataset_path, "train_processed.csv"), index=False)
    val_df.to_csv(os.path.join(dataset_path, "valid_processed.csv"), index=False)
    test_df.to_csv(os.path.join(dataset_path, "test_processed.csv"), index=False)

    print("Saved preprocessed CSV files")

    # Update config to use processed files
    config["data"]["train_csv"] = "train_processed.csv"
    config["data"]["valid_csv"] = "valid_processed.csv"
    config["data"]["test_csv"] = "test_processed.csv"

    # Save updated config
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print("Updated config file with processed CSV paths")

    # Generate statistics
    print("\nDataset Statistics:")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    # Disease distribution
    print("\nDisease Distribution (Train):")
    for col in disease_columns:
        if col in train_df.columns:
            positive = train_df[col].sum()
            percent = 100 * positive / len(train_df)
            print(f"{col}: {positive} ({percent:.2f}%)")

    print("\nPreprocessing complete!")


def create_pair_statistics(config_path):
    """Analyze the potential pairs for Siamese network training"""
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    dataset_path = config["data"]["dataset_path"]
    train_csv = os.path.join(dataset_path, config["data"]["train_csv"])

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV not found at {train_csv}")

    # Read CSV
    df = pd.read_csv(train_csv)

    # Define disease columns
    disease_columns = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    disease_columns = [col for col in disease_columns if col in df.columns]

    # Group by patient
    patient_groups = df.groupby("PatientID")

    # Analyze potential pairs
    print("\nAnalyzing potential Siamese pairs...")

    total_patients = len(patient_groups)
    patients_with_multiple_samples = 0
    total_positive_pairs = 0
    total_negative_pairs = 0

    for name, group in tqdm(patient_groups, total=len(patient_groups)):
        if len(group) > 1:
            patients_with_multiple_samples += 1

            # Count possible pairs within this patient
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    row_i = group.iloc[i][disease_columns].values
                    row_j = group.iloc[j][disease_columns].values

                    # If disease patterns match exactly
                    if np.array_equal(row_i, row_j):
                        total_positive_pairs += 1
                    else:
                        total_negative_pairs += 1

    print(
        f"Patients with multiple samples: {patients_with_multiple_samples}/{total_patients} ({100 * patients_with_multiple_samples / total_patients:.2f}%)"
    )
    print(f"Potential positive pairs (same disease pattern): {total_positive_pairs}")
    print(
        f"Potential negative pairs (different disease pattern): {total_negative_pairs}"
    )
    print(
        f"Positive-to-negative pair ratio: {total_positive_pairs / (total_positive_pairs + total_negative_pairs):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess CheXpert dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="../config/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    process_dataset(args.config)
    create_pair_statistics(args.config)
