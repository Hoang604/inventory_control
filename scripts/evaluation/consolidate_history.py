import os
import pandas as pd
import argparse
from pathlib import Path

def consolidate_history(experiment_id, output_path="paper/artifacts/training_history.csv"):
    """
    Merges Q/V and Actor diagnostics for a given experiment into a single CSV.
    """
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "logs" / experiment_id
    
    qv_path = log_dir / "training_diagnostics_qv.csv"
    actor_path = log_dir / "training_diagnostics_actor.csv"
    
    if not qv_path.exists():
        print(f"Error: Q/V diagnostics not found at {qv_path}")
        return False
    
    if not actor_path.exists():
        print(f"Error: Actor diagnostics not found at {actor_path}")
        return False
    
    print(f"Reading diagnostics from {log_dir}...")
    df_qv = pd.read_csv(qv_path)
    df_actor = pd.read_csv(actor_path)
    
    # Rename 'lr' columns to avoid collision and clarify source
    df_qv = df_qv.rename(columns={'lr': 'lr_qv'})
    df_actor = df_actor.rename(columns={'lr': 'lr_actor'})
    
    # Merge on epoch
    # We use an outer join just in case one training phase had more epochs than the other
    df_combined = pd.merge(df_qv, df_actor, on='epoch', how='outer')
    
    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df_combined.to_csv(output_file, index=False)
    print(f"Successfully consolidated history to {output_file}")
    print(f"Total epochs: {len(df_combined)}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate IQL training diagnostics into a single history file.")
    parser.add_argument("--experiment_id", type=str, required=True, help="The ID of the experiment (folder name in logs/)")
    parser.add_argument("--output", type=str, default="paper/artifacts/training_history.csv", help="Path to save the consolidated CSV")
    
    args = parser.parse_args()
    consolidate_history(args.experiment_id, args.output)
