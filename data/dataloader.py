import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
from tqdm import tqdm


class CheXpertDataset(Dataset):
    """Dataset class for loading CheXpert images"""

    def __init__(
        self, csv_file, root_dir, transform=None, target_diseases=None, use_cache=False
    ):
        """
        Args:
            csv_file: Path to the CSV file with annotations
            root_dir: Directory with all the images
            transform: Optional transform to be applied on a sample
            target_diseases: List of diseases to use as targets (if None, use all)
            use_cache: Whether to use cached tensors if available
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.use_cache = use_cache

        # If using cache, check if we have a cache CSV
        if use_cache and "cache" in os.path.basename(csv_file):
            self.use_cached_tensors = True
            print("Using cached tensors from CSV")
        else:
            self.use_cached_tensors = False

        # Print column names to debug
        print(f"CSV columns: {self.data_frame.columns.tolist()}")

        # Add PatientID column if not present
        if "PatientID" not in self.data_frame.columns:
            self.data_frame["PatientID"] = self.data_frame["Path"].apply(
                lambda x: x.split("/")[2] if len(x.split("/")) > 2 else "unknown"
            )

        # Define the diseases of interest or use all
        if target_diseases is None:
            # Get all numeric columns - safer than assuming position
            numeric_cols = self.data_frame.select_dtypes(
                include=np.number
            ).columns.tolist()
            # Filter out any non-disease columns that might be numeric
            skip_cols = ["Sex", "Age", "Frontal/Lateral", "AP/PA"]
            self.target_diseases = [col for col in numeric_cols if col not in skip_cols]

            if not self.target_diseases:
                # Fallback to known disease columns
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
                self.target_diseases = [
                    col for col in disease_columns if col in self.data_frame.columns
                ]

            print(f"Using target diseases: {self.target_diseases}")
        else:
            self.target_diseases = target_diseases

        # Check if target disease columns exist
        for disease in self.target_diseases:
            if disease not in self.data_frame.columns:
                raise ValueError(f"Disease column '{disease}' not found in CSV")

        # Ensure all target columns contain numeric data
        for disease in self.target_diseases:
            try:
                self.data_frame[disease] = pd.to_numeric(
                    self.data_frame[disease], errors="coerce"
                )
            except Exception as e:
                print(f"Error converting column {disease} to numeric: {e}")
                # Check column content
                print(f"Column values: {self.data_frame[disease].head()}")

        # Fill NaN values with 0 (uncertain->negative)
        self.data_frame[self.target_diseases] = self.data_frame[
            self.target_diseases
        ].fillna(0)

        # Convert -1 (uncertain) to 1 (positive) for the target diseases
        for disease in self.target_diseases:
            self.data_frame[disease] = self.data_frame[disease].replace(-1, 1)

    def __len__(self):
        return len(self.data_frame)

    def get_all_labels(self):
        return self.data_frame[self.target_diseases].values

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Check if we're using cached tensors
        if self.use_cached_tensors and "cached_path" in self.data_frame.columns:
            # Load the cached tensor directly
            cache_path = self.data_frame.iloc[idx]["cached_path"]
            try:
                image = torch.load(cache_path)
            except Exception as e:
                print(f"Error loading cached tensor at {cache_path}: {e}")
                raise FileNotFoundError(
                    f"Could not load cached tensor at {cache_path}. Original error: {str(e)}"
                )
        else:
            # Get the image path from the DataFrame and load from disk
            img_path = self.data_frame.iloc[idx, 0]  # Assumes first column is Path

            # Remove any duplicate "CheXpert-v1.0-small" in the path
            if "CheXpert-v1.0-small/CheXpert-v1.0-small" in img_path:
                img_path = img_path.replace(
                    "CheXpert-v1.0-small/CheXpert-v1.0-small", "CheXpert-v1.0-small"
                )

            # Join with root directory - handle paths that might have CheXpert-v1.0-small already
            if "CheXpert-v1.0-small" in self.root_dir and img_path.startswith(
                "CheXpert-v1.0-small"
            ):
                # Remove redundant prefix if the root already contains it
                img_path = img_path.replace("CheXpert-v1.0-small/", "")

            full_path = os.path.join(self.root_dir, img_path)

            try:
                # Try to load and convert the image
                image = Image.open(full_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
            except Exception as e:
                print(f"Error loading image at {full_path}: {e}")
                # Provide more details about the error
                raise FileNotFoundError(
                    f"Could not load image at {full_path}. Original error: {str(e)}"
                )

        # Convert disease labels to tensor - with extra error handling
        try:
            labels = torch.tensor(
                self.data_frame.iloc[idx][self.target_diseases].values.astype(
                    np.float32
                )
            )
        except ValueError as e:
            print(f"Error converting labels to float for index {idx}")
            print(
                f"Label values: {self.data_frame.iloc[idx][self.target_diseases].values}"
            )
            print(
                f"Data types: {self.data_frame.iloc[idx][self.target_diseases].dtypes}"
            )
            raise

        # Get patient ID from the PatientID column
        patient_id = self.data_frame.iloc[idx]["PatientID"]

        return image, labels, patient_id


class CachedDataset(Dataset):
    """Dataset that loads preprocessed image tensors directly from cache"""

    def __init__(self, csv_file, target_diseases=None):
        """
        Args:
            csv_file: Path to CSV file with cached tensor paths
            target_diseases: List of diseases to use as targets
        """
        self.data_frame = pd.read_csv(csv_file)

        if "cached_path" not in self.data_frame.columns:
            raise ValueError("CSV file does not contain cached_path column")

        # Define the diseases of interest or use all
        if target_diseases is None:
            # Get all numeric columns
            numeric_cols = self.data_frame.select_dtypes(
                include=np.number
            ).columns.tolist()
            # Filter out any non-disease columns that might be numeric
            skip_cols = ["Sex", "Age", "Frontal/Lateral", "AP/PA"]
            self.target_diseases = [col for col in numeric_cols if col not in skip_cols]

            if not self.target_diseases:
                # Fallback to known disease columns
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
                self.target_diseases = [
                    col for col in disease_columns if col in self.data_frame.columns
                ]

            print(f"Using target diseases: {self.target_diseases}")
        else:
            self.target_diseases = target_diseases

        # Add PatientID column if not present
        if "PatientID" not in self.data_frame.columns:
            self.data_frame["PatientID"] = self.data_frame["Path"].apply(
                lambda x: x.split("/")[2] if len(x.split("/")) > 2 else "unknown"
            )

        # Fill NaN values with 0 (uncertain->negative)
        self.data_frame[self.target_diseases] = self.data_frame[
            self.target_diseases
        ].fillna(0)

        # Convert -1 (uncertain) to 1 (positive) for the target diseases
        for disease in self.target_diseases:
            self.data_frame[disease] = self.data_frame[disease].replace(-1, 1)

    def get_all_labels(self):
        """Return all labels in the dataset"""
        return self.data_frame[self.target_diseases].values

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the cached tensor
        cache_path = self.data_frame.iloc[idx]["cached_path"]
        try:
            image = torch.load(cache_path)
        except Exception as e:
            print(f"Error loading cached tensor at {cache_path}: {e}")
            raise FileNotFoundError(f"Could not load cached tensor at {cache_path}")

        # Get labels
        labels = torch.tensor(
            self.data_frame.iloc[idx][self.target_diseases].values.astype(np.float32)
        )

        # Get patient ID
        patient_id = self.data_frame.iloc[idx]["PatientID"]

        return image, labels, patient_id


class SiamesePairDataset(Dataset):
    """Dataset for creating pairs for Siamese network training"""

    def __init__(self, dataset, pair_ratio=0.5):
        """
        Args:
            dataset: Base dataset (CheXpertDataset)
            pair_ratio: Ratio of positive pairs (same class) to all pairs
        """
        self.dataset = dataset
        self.pair_ratio = pair_ratio
        self.indices = list(range(len(dataset)))

        # Pre-compute indices for each label pattern
        print("Pre-computing label indices for faster pair selection...")
        self.label_to_indices = {}

        # Process all samples to build the index
        for idx in tqdm(self.indices):
            _, label, _ = self.dataset[idx]
            # Convert tensor to tuple for hashability
            label_key = tuple(label.cpu().numpy().astype(bool).tolist())

            if label_key not in self.label_to_indices:
                self.label_to_indices[label_key] = []
            self.label_to_indices[label_key].append(idx)

        print(f"Found {len(self.label_to_indices)} unique label patterns")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the first image and its label
        img1, label1, patient1 = self.dataset[idx]

        # Convert label to hashable format
        label_key = tuple(label1.cpu().numpy().astype(bool).tolist())

        # Decide whether to create a positive or negative pair
        is_positive = random.random() < self.pair_ratio

        if is_positive:
            # Get samples with the same label pattern
            same_label_indices = [
                i for i in self.label_to_indices[label_key] if i != idx
            ]
            if not same_label_indices:
                is_positive = False
                # Get a different label pattern
                diff_label_keys = [
                    k for k in self.label_to_indices.keys() if k != label_key
                ]
                if diff_label_keys:
                    random_key = random.choice(diff_label_keys)
                    pair_idx = random.choice(self.label_to_indices[random_key])
                else:
                    pair_idx = idx
            else:
                pair_idx = random.choice(same_label_indices)

        if not is_positive:
            # Get samples with different label patterns
            diff_label_keys = [
                k for k in self.label_to_indices.keys() if k != label_key
            ]
            if diff_label_keys:
                random_key = random.choice(diff_label_keys)
                pair_idx = random.choice(self.label_to_indices[random_key])
            else:
                pair_idx = idx

        # Get the second image and its label
        img2, label2, patient2 = self.dataset[pair_idx]

        # Target is 1 for similar pairs, 0 for dissimilar
        target = torch.tensor(1.0) if is_positive else torch.tensor(0.0)

        return img1, img2, target


def get_data_loaders(config):
    """Create and return data loaders for training and validation"""
    # Check if we should use cache
    use_cache = config["data"].get("use_cache", False)
    cache_dir = config["data"].get("cache_dir", None)

    if use_cache and cache_dir:
        # Check for .npy files first (memory mapping)
        train_csv = os.path.join(cache_dir, "train_cache.csv")
        val_csv = os.path.join(cache_dir, "valid_cache.csv")

        if not os.path.exists(train_csv) or not os.path.exists(val_csv):
            print("Cache CSVs not found. Falling back to regular loading.")
            use_cache = False
        else:
            # Create datasets from cached tensors
            train_dataset = CachedDataset(train_csv)
            val_dataset = CachedDataset(val_csv)

            # Create Siamese pair datasets
            train_pairs = SiamesePairDataset(train_dataset)
            val_pairs = SiamesePairDataset(val_dataset)

            # Create data loaders
            train_loader = DataLoader(
                train_pairs,
                batch_size=config["data"]["batch_size"],
                shuffle=True,
                num_workers=config["data"]["num_workers"],
                # pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )

            val_loader = DataLoader(
                val_pairs,
                batch_size=config["data"]["batch_size"],
                shuffle=False,
                num_workers=config["data"]["num_workers"],
                # pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )

            return train_loader, val_loader

    if not use_cache:
        # Standard data loading path
        # Get path from config
        base_path = config["data"]["dataset_path"]

        # Make sure there's no duplicate 'CheXpert-v1.0-small' in the path
        if base_path.endswith("CheXpert-v1.0-small/"):
            # This is the actual root directory containing the train and valid folders
            root_dir = base_path
            # CSV files are likely in the same directory
            csv_dir = base_path
        else:
            # The base path is the parent directory of CheXpert-v1.0-small
            root_dir = os.path.join(base_path, "CheXpert-v1.0-small")
            csv_dir = base_path  # Assume CSVs are in the parent directory

        # Check if the path exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory {root_dir} does not exist")

        # File paths for CSVs
        train_csv = os.path.join(csv_dir, config["data"]["train_csv"])
        val_csv = os.path.join(csv_dir, config["data"]["valid_csv"])

        # Check if CSV files exist
        if not os.path.exists(train_csv):
            # Try looking in the root_dir instead
            alt_train_csv = os.path.join(root_dir, config["data"]["train_csv"])
            if os.path.exists(alt_train_csv):
                train_csv = alt_train_csv
                val_csv = os.path.join(root_dir, config["data"]["valid_csv"])
            else:
                raise FileNotFoundError(
                    f"Could not find training CSV file at {train_csv} or {alt_train_csv}"
                )

        print(f"Using root directory: {root_dir}")
        print(f"Using CSV directory: {csv_dir}")
        print(f"Training CSV: {train_csv}")
        print(f"Validation CSV: {val_csv}")

        # Define transformations
        train_transform = transforms.Compose(
            [
                transforms.Resize(config["data"]["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize(config["data"]["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Create datasets
        train_dataset = CheXpertDataset(train_csv, root_dir, transform=train_transform)
        val_dataset = CheXpertDataset(val_csv, root_dir, transform=val_transform)

        # Create Siamese pair datasets
        train_pairs = SiamesePairDataset(train_dataset)
        val_pairs = SiamesePairDataset(val_dataset)

        # Create data loaders
        train_loader = DataLoader(
            train_pairs,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"],
            # pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        val_loader = DataLoader(
            val_pairs,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            # pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        return train_loader, val_loader
