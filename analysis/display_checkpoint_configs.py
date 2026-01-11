import os
from pathlib import Path

import torch
import os
import glob
import pprint

def load_and_print_config(file_path):
    print(f"\n{'='*80}")
    print(f"Checkpoint: {file_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(file_path):
        print("File not found.")
        return

    try:
        checkpoint = torch.load(file_path, map_location='cpu')
        if 'config' in checkpoint:
            print("Config found in checkpoint:")
            pprint.pprint(checkpoint['config'])
        else:
            print("No 'config' key found in checkpoint data.")
            print("Keys found:", list(checkpoint.keys()))
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

def main():
    if len(sys.argv) > 1:
        files = []
        for arg in sys.argv[1:]:
            matched = glob.glob(arg)
            if matched:
                files.extend(matched)
            else:
                files.append(arg)
        
        files = sorted(list(set(files)))
        
        if not files:
            print("No files matched the provided patterns.")
            return
            
        for f in files:
            load_and_print_config(f)
    else:
        print("No file paths provided. Scanning default locations (checkpoints/EXP_*/actor/*.pth)...")
        base_dir = "checkpoints"
        pattern = os.path.join(base_dir, "EXP_*", "actor", "checkpoint_epoch_*.pth")
        all_checkpoints = sorted(glob.glob(pattern))
        
        experiments = {}
        for cp in all_checkpoints:
            parts = cp.split(os.sep)
            if len(parts) >= 2:
                exp_name = parts[1]
                if exp_name not in experiments:
                     experiments[exp_name] = cp
        
        if not experiments:
            print("No checkpoints found in default location.")
            return

        print(f"Found {len(experiments)} experiments. Displaying config for one checkpoint each.")
        for exp_name in sorted(experiments.keys()):
            load_and_print_config(experiments[exp_name])

if __name__ == "__main__":
    main()
