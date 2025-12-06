import pandas as pd
import numpy as np
import sys
import json
import os
from pathlib import Path

def ascii_plot(values, height=10, width=50, title="Plot"):
    """Generates a simple ASCII plot for terminal visualization."""
    if not values:
        return "No Data"
    
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val != min_val else 1
    
    # Normalize values to fit height
    normalized = [int((v - min_val) / range_val * (height - 1)) for v in values]
    
    # Downsample to width
    if len(normalized) > width:
        indices = np.linspace(0, len(normalized) - 1, width).astype(int)
        plot_data = [normalized[i] for i in indices]
    else:
        plot_data = normalized

    # Create grid
    grid = [[' ' for _ in range(len(plot_data))] for _ in range(height)]
    
    for x, y in enumerate(plot_data):
        grid[height - 1 - y][x] = '*'

    # Build string
    output = [f"\n--- {title} ---"]
    output.append(f"Max: {max_val:.2f}")
    for row in grid:
        output.append("".join(row))
    output.append(f"Min: {min_val:.2f}")
    return "\n".join(output)

def calculate_trend_slope(series):
    """Calculates the linear slope of a series."""
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    slope, _ = np.polyfit(x, series, 1)
    return slope

def analyze_diagnostics(file_path):
    if not os.path.exists(file_path):
        print(json.dumps({"error": "File not found", "path": file_path}))
        return

    df = pd.read_csv(file_path)
    
    # --- METRICS EXTRACTION ---
    epochs = df['epoch'].tolist()
    q_values = df['avg_q_val'].tolist() if 'avg_q_val' in df else []
    v_values = df['avg_v_val'].tolist() if 'avg_v_val' in df else []
    entropies = df['avg_entropy'].tolist() if 'avg_entropy' in df else []
    advantages = df['avg_advantage'].tolist() if 'avg_advantage' in df else []
    
    # --- TECHNICAL ANALYSIS (MACHINE READABLE) ---
    q_slope = calculate_trend_slope(q_values)
    entropy_slope = calculate_trend_slope(entropies)
    
    # Check for Q-Value Explosion (More robust heuristic)
    q_explosion = False
    if q_values and len(q_values) > 1:
        # If final Q-value is > 400 (roughly 2x max observed reward)
        # Or if it increased significantly from a non-trivial start
        if q_values[-1] > 400.0: # Absolute threshold based on test results
            q_explosion = True
        elif q_values[0] != 0 and abs(q_values[-1] / q_values[0]) > 5.0 and q_values[0] > 1.0: # 5x increase from a non-zero start
             q_explosion = True
        elif q_values[0] <= 1.0 and q_values[-1] > 100.0: # From near zero to large positive
            q_explosion = True
    
    # Check for Mode Collapse (Entropy < threshold)
    min_entropy = min(entropies) if entropies else 0
    mode_collapse = min_entropy < -5.0 # Heuristic threshold for continuous control
    
    # Divergence Check (Advantage growing?)
    adv_slope = calculate_trend_slope(advantages)

    machine_report = {
        "status": "success",
        "metrics": {
            "q_trend_slope": q_slope,
            "q_explosion_flag": q_explosion,
            "entropy_trend_slope": entropy_slope,
            "min_entropy": min_entropy,
            "advantage_trend_slope": adv_slope
        },
        "flags": {
            "q_value_explosion": q_explosion,
            "mode_collapse": bool(mode_collapse),
            "value_divergence": bool(adv_slope > 0.1) # Threshold for growing advantage
        },
        "summary": ""
    }

    # --- HUMAN READABLE REPORT ---
    print("="*60)
    print("IQL TRAINING DIAGNOSTICS REPORT")
    print("="*60)
    
    print(f"Log File: {file_path}")
    print(f"Total Epochs: {len(epochs)}\n")
    
    # Q-Value Analysis
    print("1. VALUE ESTIMATION (Q-Values)")
    if q_values:
        print(f"   Start: {q_values[0]:.2f} | End: {q_values[-1]:.2f}")
        print(f"   Trend: {'RISING' if q_slope > 0 else 'FALLING'} (Slope: {q_slope:.4f})")
        if machine_report['flags']['q_value_explosion']:
            print("   [WARNING] Q-Values are exploding! This indicates divergence.")
        else:
            print("   [OK] Q-Values appear stable.")
        print(ascii_plot(q_values, title="Avg Q-Value History"))
    else:
        print("   [N/A] No Q-Value data found.")
        
    print("\n" + "-"*40 + "\n")
    
    # V-Value Analysis
    print("2. VALUE ESTIMATION (V-Values)")
    if v_values:
        v_slope = calculate_trend_slope(v_values)
        print(f"   Start: {v_values[0]:.2f} | End: {v_values[-1]:.2f}")
        print(f"   Trend: {'RISING' if v_slope > 0 else 'FALLING'} (Slope: {v_slope:.4f})")
        print(ascii_plot(v_values, title="Avg V-Value History"))
    else:
        print("   [N/A] No V-Value data found.")

    print("\n" + "-"*40 + "\n")

    # Policy Entropy Analysis
    print("3. POLICY DIVERSITY (Entropy)")
    if entropies:
        print(f"   Start: {entropies[0]:.2f} | End: {entropies[-1]:.2f}")
        print(f"   Trend: {'INCREASING' if entropy_slope > 0 else 'DECREASING'}")
        if machine_report['flags']['mode_collapse']:
            print("   [WARNING] Entropy is extremely low. The policy has likely collapsed to a single deterministic action.")
        else:
            print("   [OK] Policy maintains some stochasticity.")
        print(ascii_plot(entropies, title="Avg Entropy History"))
    else:
        print("   [N/A] No Entropy data found.")
        
    print("\n" + "-"*40 + "\n")
    
    # Final Conclusion for User
    print("4. SUMMARY & RECOMMENDATION")
    if machine_report['flags']['q_value_explosion']:
        summary = "CRITICAL FAILURE: Value Divergence. The critic is overestimating rewards, causing the actor to chase fake high-value actions."
        rec = "FIX: Scale down rewards (e.g., reward / 10.0), decrease Discount Factor (gamma), or increase Expectile (tau)."
    elif machine_report['flags']['mode_collapse']:
        summary = "FAILURE: Mode Collapse. The agent has stopped exploring and is stuck repeating one action."
        rec = "FIX: Increase Actor regularization (beta) or check if Q-values are forcing extreme actions."
    elif q_slope < -0.5:
        summary = "WARNING: Value Collapse. The agent thinks everything leads to failure."
        rec = "FIX: Check reward function. Is the agent never seeing positive rewards?"
    else:
        summary = "STABLE: Training looks healthy mathematically."
        rec = "INFO: If performance is still bad, the issue is likely 'Out of Distribution' actions (IQL Gap). Use Early Stopping."
    
    print(f"   Diagnosis: {summary}")
    print(f"   Advice:    {rec}")
    print("="*60)
    
    # Output JSON for Agent (Hidden/File)
    machine_report['summary'] = summary
    with open("diagnostics_machine.json", "w") as f:
        json.dump(machine_report, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_diagnostics.py <path_to_csv>")
        # Try to find the latest log automatically if not provided
        # This is a helper for the user
        log_dir = Path("logs")
        if log_dir.exists():
            # Find latest experiment folder
            experiments = sorted([d for d in log_dir.iterdir() if d.is_dir()], key=os.path.getmtime, reverse=True)
            if experiments:
                latest_csv = experiments[0] / "training_diagnostics.csv"
                if latest_csv.exists():
                    print(f"Auto-detected latest log: {latest_csv}\n")
                    analyze_diagnostics(str(latest_csv))
                else:
                    print("No diagnostics file found in latest experiment.")
    else:
        analyze_diagnostics(sys.argv[1])
