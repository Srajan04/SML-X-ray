import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import glob
import yaml

def convert_pt_to_npy(config_path="config/config.yaml", output_dir=None):
    """Convert PyTorch .pt files to NumPy .npy files for memory mapping"""
    # Load configuration to get cache directory
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    cache_dir = config["data"].get("cache_dir", None)
    if not cache_dir:
        raise ValueError("Cache directory not specified in config")
    
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"Cache directory {cache_dir} not found")
        
    # If output directory not specified, create a new one
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(cache_dir), "npy_cache")
    
    print(f"Converting tensors from {cache_dir} to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files in the cache directory
    csv_files = glob.glob(os.path.join(cache_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {cache_dir}")
        return
    
    for csv_file in csv_files:
        print(f"Processing CSV file: {os.path.basename(csv_file)}")
        
        try:
            df = pd.read_csv(csv_file)
            
            if 'cached_path' not in df.columns:
                print(f"CSV file {csv_file} does not have 'cached_path' column")
                continue
                
            # New paths for updated CSV
            new_paths = []
            
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Converting files"):
                pt_path = row['cached_path']
                
                if not os.path.exists(pt_path):
                    print(f"Warning: File not found: {pt_path}")
                    new_paths.append(pt_path)  # Keep original path in case of error
                    continue
                
                if not pt_path.endswith('.pt'):
                    print(f"Warning: Not a .pt file: {pt_path}")
                    new_paths.append(pt_path)  # Keep original path if not a .pt file
                    continue
                    
                # Create output path with .npy extension
                filename = os.path.basename(pt_path).replace('.pt', '.npy')
                npy_path = os.path.join(output_dir, filename)
                
                # Convert and save if not already exists
                if not os.path.exists(npy_path):
                    try:
                        tensor = torch.load(pt_path)
                        np.save(npy_path, tensor.numpy())
                    except Exception as e:
                        print(f"Error converting {pt_path}: {e}")
                        npy_path = pt_path  # Keep original path in case of error
                
                new_paths.append(npy_path)
            
            # Update CSV with new paths
            df['cached_path'] = new_paths
            
            # Save updated CSV
            output_csv = os.path.join(output_dir, os.path.basename(csv_file))
            df.to_csv(output_csv, index=False)
            print(f"Updated CSV saved to {output_csv}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    # Create a new config file pointing to the converted cache
    new_config = config.copy()
    new_config["data"]["cache_dir"] = output_dir
    new_config_path = f"{config_path}.npy_cached"
    
    with open(new_config_path, "w") as file:
        yaml.dump(new_config, file, default_flow_style=False)
        
    print(f"Created updated config at {new_config_path}")
    print(f"To use the memory-mapped tensors, run with: --config {new_config_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert PyTorch tensors to NumPy arrays for memory mapping")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for NumPy arrays (optional)")
    
    args = parser.parse_args()
    
    convert_pt_to_npy(args.config, args.output_dir)