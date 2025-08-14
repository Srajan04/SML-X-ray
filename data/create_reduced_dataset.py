import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def create_reduced_dataset(config_path, reduction_factor=0.4, use_balanced=True):
    # Load config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Determine input train CSV path
    dataset_path = config["data"]["dataset_path"]
    train_csv = (
        os.path.join(dataset_path, "train_balanced.csv")
        if use_balanced
        else config["data"]["train_csv"]
    )

    if not os.path.exists(train_csv):
        print(f"Input file {train_csv} not found. Run create_balanced_split.py first.")
        return None

    # Load train csv
    df = pd.read_csv(train_csv)

    # Get patient IDs if they exist
    if "PatientID" not in df.columns:
        df["PatientID"] = df["Path"].apply(
            lambda x: x.split("/")[2] if len(x.split("/")) > 2 else "unknown"
        )

    # Stratify by patient's disease patterns if possible
    patients = df["PatientID"].unique()
    sampled_patients, _ = train_test_split(
        patients,
        train_size=reduction_factor,
        random_state=42,
        stratify=None,  # Replace with stratification if available
    )

    # Filter the dataframe
    reduced_df = df[df["PatientID"].isin(sampled_patients)]

    # Save the reduced dataset
    reduced_csv = os.path.join(
        dataset_path, f"train_balanced_reduced_{int(reduction_factor * 100)}pct.csv"
    )
    reduced_df.to_csv(reduced_csv, index=False)

    print(f"Original dataset: {len(df)} samples")
    print(f"Reduced dataset: {len(reduced_df)} samples ({reduction_factor * 100:.0f}%)")
    print(f"Saved to: {reduced_csv}")

    return reduced_csv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config"
    )
    parser.add_argument(
        "--factor", type=float, default=0.4, help="Reduction factor (0.0-1.0)"
    )
    args = parser.parse_args()

    create_reduced_dataset(args.config, args.factor)
