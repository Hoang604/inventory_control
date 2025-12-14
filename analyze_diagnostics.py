import pandas as pd
import os
import matplotlib.pyplot as plt

def analyze_diagnostics(file_path):
    """
    Analyzes the training diagnostics CSV file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Analyzing: {file_path} ---")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 1. Raw Data Preview
    print("\n[Raw Data Preview (First 5 rows)]")
    print(df.head().to_string())
    print("\n[Raw Data Preview (Last 5 rows)]")
    print(df.tail().to_string())

    # 2. Statistical Summary
    print("\n[Statistical Summary]")
    stats = df.describe().transpose()
    print(stats[['mean', 'std', 'min', '50%', 'max']].to_string())

    # 3. Trend Analysis (Stability Check)
    print("\n[Trend Analysis]")
    
    # Check if Loss increased in the second half
    half_epoch = len(df) // 2
    first_half = df.iloc[:half_epoch]
    second_half = df.iloc[half_epoch:]

    metrics_to_check = ['val_q_loss', 'val_v_loss', 'val_q_std']
    
    for metric in metrics_to_check:
        if metric in df.columns:
            mean_1 = first_half[metric].mean()
            mean_2 = second_half[metric].mean()
            change = ((mean_2 - mean_1) / mean_1) * 100
            
            print(f"{metric}:")
            print(f"  First Half Mean: {mean_1:.4f}")
            print(f"  Second Half Mean: {mean_2:.4f}")
            print(f"  Change: {change:+.2f}%")
            
            if change > 20:
                print(f"  -> WARNING: Significant increase detected! Possible overfitting/instability.")
            elif change < -20:
                print(f"  -> Good: Metric significantly decreased.")
            else:
                print(f"  -> Stable: Metric remained relatively constant.")
    
    # 4. Best Epoch Identification
    if 'val_q_loss' in df.columns:
        best_epoch_row = df.loc[df['val_q_loss'].idxmin()]
        print(f"\n[Best Epoch by Min Validation Q-Loss]")
        print(f"Epoch: {int(best_epoch_row['epoch'])}")
        print(f"Val Q-Loss: {best_epoch_row['val_q_loss']:.4f}")
        if 'val_q_std' in df.columns:
            print(f"Val Q-Std:  {best_epoch_row['val_q_std']:.4f}")

if __name__ == "__main__":
    # Analyzing the BAD run
    TARGET_FILE = "logs/inv_management_iql_minmax_run_06122025_234052/training_diagnostics_qv.csv"
    
    analyze_diagnostics(TARGET_FILE)
