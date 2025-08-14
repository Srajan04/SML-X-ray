import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def process_label(col_name, value):
    """Convert label value to float. For categorical columns, map to numeric values."""
    # Handle special case columns that need string-to-number conversion
    if col_name == "Sex":
        if isinstance(value, str):
            return 0.0 if value.lower() == "male" else 1.0
        return float(value)
    
    # Handle "Frontal/Lateral" if needed
    elif col_name == "Frontal/Lateral":
        if isinstance(value, str):
            return 0.0 if value.lower() == "frontal" else 1.0
        return float(value)
        
    # Handle "AP/PA" if needed
    elif col_name == "AP/PA":
        if isinstance(value, str):
            return 0.0 if value.lower() == "ap" else 1.0
        return float(value)
        
    # For all other columns, try direct conversion
    try:
        return float(value)
    except Exception as e:
        # If conversion fails, provide informative error
        raise ValueError(f"Could not convert value '{value}' of type {type(value)} in column '{col_name}' to float.") from e

class MemoryMappedDataset(Dataset):
    """Dataset using memory-mapped cached tensors for efficient loading"""
    def __init__(self, csv_file, target_diseases=None):
        self.data_frame = pd.read_csv(csv_file)
        if target_diseases is None:
            # Use all columns except known non-targets
            exclude = ["Path", "cached_path", "PatientID", "StudyID", "StudyDate"]
            self.target_diseases = [col for col in self.data_frame.columns if col not in exclude]
        else:
            self.target_diseases = target_diseases

        print(f"Using target diseases: {self.target_diseases}")
        self.mmap_files = {}
        self.patient_ids = self.data_frame['PatientID'].tolist() if "PatientID" in self.data_frame.columns else None

        first_path = self.data_frame.iloc[0]['cached_path']
        if not os.path.exists(first_path):
            raise FileNotFoundError(f"Cached tensor not found at {first_path}")

        self.file_ext = os.path.splitext(first_path)[1]
        if self.file_ext not in ['.npy', '.pt']:
            raise ValueError(f"Unsupported cache file format: {self.file_ext}. Expected .npy or .pt")

        if self.file_ext == '.pt':
            print("Note: Using PyTorch tensors (.pt) which will be loaded fully into memory. For true memory mapping, consider converting to .npy format.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cache_path = self.data_frame.iloc[idx]['cached_path']
        # Load image
        if self.file_ext == '.npy':
            if cache_path not in self.mmap_files:
                try:
                    self.mmap_files[cache_path] = np.load(cache_path, mmap_mode='r')
                except Exception as e:
                    print(f"Error memory-mapping {cache_path}: {e}")
                    raise
            image = torch.from_numpy(self.mmap_files[cache_path]).clone()
        else:
            try:
                image = torch.load(cache_path)
            except Exception as e:
                print(f"Error loading tensor from {cache_path}: {e}")
                raise

        # Process labels column by column
        raw_labels = self.data_frame.iloc[idx][self.target_diseases]
        processed_labels = []
        for col, value in raw_labels.iteritems():
            processed_labels.append(process_label(col, value))
        labels = torch.tensor(processed_labels, dtype=torch.float32)

        patient_id = self.patient_ids[idx] if self.patient_ids is not None else idx
        return image, labels, patient_id