import os
import torch
import pandas as pd
from pathlib import Path
import yaml
import shutil
from tqdm import tqdm


def check_cache_status(config_path="config/config.yaml"):
    """
    Check if cache is available and enabled

    Args:
        config_path: Path to the configuration file

    Returns:
        bool: True if cache is available and enabled
    """
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    use_cache = config["data"].get("use_cache", False)
    cache_dir = config["data"].get("cache_dir", None)

    if not use_cache:
        print("Cache usage is disabled in config")
        return False

    if not cache_dir:
        print("Cache directory not specified in config")
        return False

    if not os.path.exists(cache_dir):
        print(f"Cache directory {cache_dir} does not exist")
        return False

    # Check if cache CSVs exist
    train_cache_csv = os.path.join(cache_dir, "train_cache.csv")
    valid_cache_csv = os.path.join(cache_dir, "valid_cache.csv")
    test_cache_csv = os.path.join(cache_dir, "test_cache.csv")

    if not os.path.exists(train_cache_csv):
        print(f"Train cache CSV not found: {train_cache_csv}")
        return False

    if not os.path.exists(valid_cache_csv):
        print(f"Validation cache CSV not found: {valid_cache_csv}")
        return False

    # Check if the CSVs actually point to cached tensors
    try:
        df = pd.read_csv(train_cache_csv)
        if "cached_path" not in df.columns:
            print("Train cache CSV does not contain cached_path column")
            return False

        # Check that a sample tensor exists
        first_cache_path = df["cached_path"].iloc[0]
        if not os.path.exists(first_cache_path):
            print(f"Cached tensor not found: {first_cache_path}")
            return False

        # Try loading a tensor to confirm it works
        torch.load(first_cache_path)

    except Exception as e:
        print(f"Error checking cache: {e}")
        return False

    print("Cache is available and ready to use")
    return True


def clear_cache(config_path="config/config.yaml", confirm=True):
    """
    Clear the cache directory

    Args:
        config_path: Path to the configuration file
        confirm: Whether to ask for confirmation

    Returns:
        bool: True if cache was cleared, False otherwise
    """
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    cache_dir = config["data"].get("cache_dir", None)

    if not cache_dir:
        print("Cache directory not specified in config")
        return False

    if not os.path.exists(cache_dir):
        print(f"Cache directory {cache_dir} does not exist")
        return False

    if confirm:
        response = input(
            f"Are you sure you want to delete all files in {cache_dir}? (y/n): "
        )
        if response.lower() != "y":
            print("Cache clearing aborted")
            return False

    # Delete everything in cache directory
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".pt"):  # Only delete tensor files
                os.unlink(os.path.join(root, f))

    # Delete cache CSVs
    for csv_file in ["train_cache.csv", "valid_cache.csv", "test_cache.csv"]:
        csv_path = os.path.join(cache_dir, csv_file)
        if os.path.exists(csv_path):
            os.unlink(csv_path)

    print(f"Cache directory {cache_dir} has been cleared")
    return True


def update_config_for_cache(config_path="config/config.yaml", use_cache=True):
    """
    Update the configuration to use cache

    Args:
        config_path: Path to the configuration file
        use_cache: Whether to enable cache usage
    """
    # Load configuration
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Get cache directory
    cache_dir = config["data"].get("cache_dir")
    if not cache_dir:
        dataset_path = config["data"]["dataset_path"]
        cache_dir = os.path.join(dataset_path, "preprocessed_cache")
        config["data"]["cache_dir"] = cache_dir

    # Update cache settings
    config["data"]["use_cache"] = use_cache

    # Save updated config
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

    print(f"Updated config: cache {'enabled' if use_cache else 'disabled'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cache utilities")
    parser.add_argument(
        "--config", type=str, default="../config/config.yaml", help="Path to config"
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["check", "clear", "enable", "disable"],
        default="check",
        help="Action to perform",
    )

    args = parser.parse_args()

    if args.action == "check":
        check_cache_status(args.config)
    elif args.action == "clear":
        clear_cache(args.config)
    elif args.action == "enable":
        update_config_for_cache(args.config, True)
    elif args.action == "disable":
        update_config_for_cache(args.config, False)
