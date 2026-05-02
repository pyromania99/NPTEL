#!/usr/bin/env python3
"""
Eigenvalue Placement Controller Analysis Script

Analyzes the C matrix evolution from flight logs:
- G1 gain matrix scaling and time-varying behavior
- c4 compensation vector magnitude
- M matrix eigenvalue stability
- A1 state-dependent dynamics correlation with yaw rate
- Control effectiveness metrics

Usage: python3 analyze_controller_matrix.py <path_to_csv_log>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

def load_log(csv_path):
    """Load flight log CSV file"""
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Duration: {df['timestamp'].iloc[-1]:.2f} seconds")
    return df

def analyze_g1_gains(df):
    """Analyze G1 gain matrix properties"""
    print("\n" + "="*60)
    print("G1 GAIN MATRIX ANALYSIS")
    print("="*60)
    
    # Extract G1 elements
    g1_cols = ['G1_11', 'G1_12', 'G1_13', 'G1_21', 'G1_22', 'G1_23', 'G1_31', 'G1_32', 'G1_33']
    
    # Statistical summary
    print("\nG1 Element Statistics:")
    print("-" * 60)
    for col in g1_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            range_val = max_val - min_val
            print(f"{col:8s}: mean={mean_val:8.3f}, std={std_val:7.3f}, "
                  f"range=[{min_val:8.3f}, {max_val:8.3f}], Δ={range_val:7.3f}")
    
    # Check third row constraint (should be near zero)
    print("\nThird Row Constraint Verification (should be ≈ 0):")
    print("-" * 60)
    for col in ['G1_31', 'G1_32', 'G1_33']:
        if col in df.columns:
            max_abs = df[col].abs().max()
            rms = np.sqrt((df[col]**2).mean())
            print(f"{col}: max_abs={max_abs:.6f}, rms={rms:.6f}")
            if max_abs > 1e-3:
                print(f"  ⚠️  WARNING: {col} exceeds threshold (> 0.001)")
    
    # Gain scaling factor analysis
    print("\nG1 Diagonal Dominance:")
    print("-" * 60)
    if all(col in df.columns for col in ['G1_11', 'G1_22']):
        g1_11_mean = df['G1_11'].mean()
        g1_22_mean = df['G1_22'].mean()
        print(f"Average G1(1,1): {g1_11_mean:.3f}")
        print(f"Average G1(2,2): {g1_22_mean:.3f}")
        print(f"Ratio G1(1,1)/G1(2,2): {g1_11_mean/g1_22_mean if g1_22_mean != 0 else np.nan:.3f}")
    
    # Time-varying behavior
    print("\nG1 Time-Varying Characteristics:")
    print("-" * 60)
    for col in ['G1_11', 'G1_12', 'G1_21', 'G1_22']:
        if col in df.columns:
            variation = df[col].std() / abs(df[col].mean()) if df[col].mean() != 0 else np.nan
            print(f"{col} coefficient of variation: {variation:.3f} ({variation*100:.1f}%)")

def analyze_c4_compensation(df):
    """Analyze c4 compensation vector"""
    print("\n" + "="*60)
    print("c4 COMPENSATION VECTOR ANALYSIS")
    print("="*60)
    
    c4_cols = ['c4_x', 'c4_y', 'c4_z']
    
    print("\nc4 Element Statistics:")
    print("-" * 60)
    for col in c4_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"{col:6s}: mean={mean_val:8.3f}, std={std_val:7.3f}, "
                  f"range=[{min_val:8.3f}, {max_val:8.3f}]")
    
    # c4 magnitude over time
    if all(col in df.columns for col in c4_cols):
        df['c4_magnitude'] = np.sqrt(df['c4_x']**2 + df['c4_y']**2 + df['c4_z']**2)
        print(f"\nc4 Vector Magnitude:")
        print(f"  Mean: {df['c4_magnitude'].mean():.3f}")
        print(f"  Max:  {df['c4_magnitude'].max():.3f}")
        print(f"  Std:  {df['c4_magnitude'].std():.3f}")

def analyze_eigenvalues(df):
    """Analyze M matrix eigenvalues stability"""
    print("\n" + "="*60)
    print("M MATRIX EIGENVALUE ANALYSIS")
    print("="*60)
    
    eig_cols = ['M_eig1', 'M_eig2', 'M_eig3']
    
    print("\nEigenvalue Statistics:")
    print("-" * 60)
    for col in eig_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"{col:8s}: mean={mean_val:8.3f}, std={std_val:7.3f}, "
                  f"range=[{min_val:8.3f}, {max_val:8.3f}]")
            
            # Check stability
            if mean_val >= 0:
                print(f"  ⚠️  WARNING: {col} is non-negative (unstable)!")
            
            # Check variability
            if std_val > 0.1 * abs(mean_val):
                print(f"  ⚠️  NOTE: {col} has high variability (>10% of mean)")

def analyze_a1_dynamics(df):
    """Analyze A1 state-dependent dynamics"""
    print("\n" + "="*60)
    print("A1 STATE-DEPENDENT DYNAMICS ANALYSIS")
    print("="*60)
    
    a1_cols = ['A1_11', 'A1_12', 'A1_21', 'A1_22']
    
    print("\nA1 Element Statistics:")
    print("-" * 60)
    for col in a1_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"{col:8s}: mean={mean_val:8.3f}, std={std_val:7.3f}, "
                  f"range=[{min_val:8.3f}, {max_val:8.3f}]")
    
    # Correlation with yaw rate
    print("\nCorrelation with Yaw Rate (r):")
    print("-" * 60)
    if 'r' in df.columns:
        for col in ['A1_12', 'A1_21']:
            if col in df.columns:
                corr = df[col].corr(df['r'])
                print(f"{col} vs r: correlation = {corr:.4f}")
                
                # A1_12 should equal r, A1_21 should equal -r
                if col == 'A1_12':
                    diff = (df[col] - df['r']).abs().mean()
                    print(f"  Mean |A1_12 - r|: {diff:.6f} (should be ≈ 0)")
                elif col == 'A1_21':
                    diff = (df[col] + df['r']).abs().mean()
                    print(f"  Mean |A1_21 + r|: {diff:.6f} (should be ≈ 0)")

def analyze_m_matrix(df):
    """Analyze M matrix elements"""
    print("\n" + "="*60)
    print("M MATRIX ELEMENT ANALYSIS")
    print("="*60)
    
    m_cols = ['M_11', 'M_12', 'M_21', 'M_22']
    
    print("\nM Matrix Statistics:")
    print("-" * 60)
    for col in m_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"{col:8s}: mean={mean_val:8.3f}, std={std_val:7.3f}, "
                  f"range=[{min_val:8.3f}, {max_val:8.3f}]")

def analyze_control_effectiveness(df):
    """Analyze control effectiveness metrics"""
    print("\n" + "="*60)
    print("CONTROL EFFECTIVENESS ANALYSIS")
    print("="*60)
    
    # Torque command statistics
    print("\nTorque Commands:")
    print("-" * 60)
    for axis in ['x', 'y', 'z']:
        col = f'torque_{axis}'
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            max_abs = df[col].abs().max()
            print(f"torque_{axis}: mean={mean_val:7.3f}, std={std_val:6.3f}, max_abs={max_abs:6.3f}")
    
    # Attitude error metrics
    print("\nAttitude Errors:")
    print("-" * 60)
    for err in ['att_err_roll', 'att_err_pitch']:
        if err in df.columns:
            rms = np.sqrt((df[err]**2).mean())
            max_abs = df[err].abs().max()
            print(f"{err}: rms={rms:.4f} rad ({np.degrees(rms):.2f}°), "
                  f"max_abs={max_abs:.4f} rad ({np.degrees(max_abs):.2f}°)")
    
    # Rate error metrics
    print("\nRate Errors:")
    print("-" * 60)
    for err in ['rate_err_p', 'rate_err_q', 'rate_err_r']:
        if err in df.columns:
            rms = np.sqrt((df[err]**2).mean())
            max_abs = df[err].abs().max()
            print(f"{err}: rms={rms:.4f} rad/s, max_abs={max_abs:.4f} rad/s")

def plot_g1_evolution(df, output_dir):
    """Plot G1 gain matrix evolution over time"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('G1 Gain Matrix Evolution Over Time', fontsize=16)
    
    g1_elements = [
        ['G1_11', 'G1_12', 'G1_13'],
        ['G1_21', 'G1_22', 'G1_23'],
        ['G1_31', 'G1_32', 'G1_33']
    ]
    
    for i in range(3):
        for j in range(3):
            col = g1_elements[i][j]
            ax = axes[i, j]
            if col in df.columns:
                ax.plot(df['timestamp'], df[col], linewidth=0.8)
                ax.set_title(f'{col}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Gain Value')
                ax.grid(True, alpha=0.3)
                
                # Highlight if third row (should be ~0)
                if i == 2:
                    ax.axhline(0, color='r', linestyle='--', linewidth=1, alpha=0.5)
                    ax.set_ylim([-0.01, 0.01])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'g1_matrix_evolution.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

def plot_c4_and_eigenvalues(df, output_dir):
    """Plot c4 compensation and eigenvalues"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # c4 components
    ax = axes[0]
    for col in ['c4_x', 'c4_y', 'c4_z']:
        if col in df.columns:
            ax.plot(df['timestamp'], df[col], label=col, linewidth=1)
    ax.set_title('c4 Compensation Vector Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('c4 Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # M eigenvalues
    ax = axes[1]
    for col in ['M_eig1', 'M_eig2', 'M_eig3']:
        if col in df.columns:
            ax.plot(df['timestamp'], df[col], label=col, linewidth=1)
    ax.axhline(0, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Stability boundary')
    ax.set_title('M Matrix Eigenvalues Over Time')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Eigenvalue')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'c4_and_eigenvalues.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

def plot_a1_vs_yaw_rate(df, output_dir):
    """Plot A1 dynamics vs yaw rate"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('A1 State-Dependent Dynamics vs Yaw Rate', fontsize=16)
    
    if 'r' not in df.columns:
        print("Warning: Yaw rate 'r' not found in data")
        plt.close()
        return
    
    # A1_12 vs r (should be equal)
    ax = axes[0, 0]
    if 'A1_12' in df.columns:
        ax.scatter(df['r'], df['A1_12'], s=1, alpha=0.5)
        ax.plot(df['r'], df['r'], 'r--', label='y=x (expected)', linewidth=2)
        ax.set_title('A1(1,2) vs Yaw Rate r')
        ax.set_xlabel('r (rad/s)')
        ax.set_ylabel('A1(1,2)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # A1_21 vs r (should be -r)
    ax = axes[0, 1]
    if 'A1_21' in df.columns:
        ax.scatter(df['r'], df['A1_21'], s=1, alpha=0.5)
        ax.plot(df['r'], -df['r'], 'r--', label='y=-x (expected)', linewidth=2)
        ax.set_title('A1(2,1) vs Yaw Rate r')
        ax.set_xlabel('r (rad/s)')
        ax.set_ylabel('A1(2,1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # A1_12 time series
    ax = axes[1, 0]
    if 'A1_12' in df.columns:
        ax.plot(df['timestamp'], df['A1_12'], label='A1(1,2)', linewidth=1)
        ax.plot(df['timestamp'], df['r'], label='r', linewidth=1, alpha=0.7)
        ax.set_title('A1(1,2) and r Over Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value (rad/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # A1_21 time series
    ax = axes[1, 1]
    if 'A1_21' in df.columns:
        ax.plot(df['timestamp'], df['A1_21'], label='A1(2,1)', linewidth=1)
        ax.plot(df['timestamp'], -df['r'], label='-r', linewidth=1, alpha=0.7)
        ax.set_title('A1(2,1) and -r Over Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value (rad/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'a1_vs_yaw_rate.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

def plot_gain_scaling_analysis(df, output_dir):
    """Analyze gain scaling factors"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Gain Scaling Factor Analysis', fontsize=16)
    
    # G1 diagonal elements over time
    ax = axes[0, 0]
    for col in ['G1_11', 'G1_22']:
        if col in df.columns:
            ax.plot(df['timestamp'], df[col], label=col, linewidth=1)
    ax.set_title('G1 Diagonal Elements')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Gain Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # G1 off-diagonal elements
    ax = axes[0, 1]
    for col in ['G1_12', 'G1_21']:
        if col in df.columns:
            ax.plot(df['timestamp'], df[col], label=col, linewidth=1)
    ax.set_title('G1 Off-Diagonal Elements')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Gain Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gain magnitude vs yaw rate
    ax = axes[1, 0]
    if all(col in df.columns for col in ['G1_11', 'r']):
        ax.scatter(df['r'].abs(), df['G1_11'].abs(), s=1, alpha=0.5)
        ax.set_title('|G1(1,1)| vs |Yaw Rate|')
        ax.set_xlabel('|r| (rad/s)')
        ax.set_ylabel('|G1(1,1)|')
        ax.grid(True, alpha=0.3)
    
    # M matrix elements
    ax = axes[1, 1]
    for col in ['M_11', 'M_22']:
        if col in df.columns:
            ax.plot(df['timestamp'], df[col], label=col, linewidth=1)
    ax.set_title('M Matrix Diagonal Elements')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('M Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'gain_scaling_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

def plot_control_performance(df, output_dir):
    """Plot control performance metrics"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Control Performance with Eigenvalue Placement', fontsize=16)
    
    # Attitude errors
    ax = axes[0]
    for err in ['att_err_roll', 'att_err_pitch']:
        if err in df.columns:
            ax.plot(df['timestamp'], np.degrees(df[err]), label=err, linewidth=1)
    ax.set_title('Attitude Errors')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (degrees)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rate errors
    ax = axes[1]
    for err in ['rate_err_p', 'rate_err_q']:
        if err in df.columns:
            ax.plot(df['timestamp'], df[err], label=err, linewidth=1)
    ax.set_title('Rate Errors')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error (rad/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Torque commands
    ax = axes[2]
    for axis in ['x', 'y', 'z']:
        col = f'torque_{axis}'
        if col in df.columns:
            ax.plot(df['timestamp'], df[col], label=f'τ_{axis}', linewidth=1)
    ax.set_title('Torque Commands')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'control_performance.png')
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()

def find_latest_log():
    """Find the most recent flight log in the default directory"""
    log_dir = "/home/pyro/ws_offboard_control/flight_data/logs"
    
    if not os.path.exists(log_dir):
        return None
    
    # Find all CSV files
    csv_files = list(Path(log_dir).glob("flight_log_*.csv"))
    
    if not csv_files:
        return None
    
    # Sort by modification time, newest first
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    return str(latest_file)

def main():
    if len(sys.argv) < 2:
        # Try to find latest log automatically
        csv_path = find_latest_log()
        if csv_path is None:
            print("Usage: python3 analyze_controller_matrix.py [path_to_csv_log]")
            print("\nNo log file specified and no logs found in flight_data/logs/")
            print("\nExample:")
            print("  python3 analyze_controller_matrix.py flight_data/logs/flight_log_20251210_143022.csv")
            sys.exit(1)
        print(f"No log specified, using latest: {os.path.basename(csv_path)}")
    else:
        csv_path = sys.argv[1]
    
    # Load data
    df = load_log(csv_path)
    
    # Create output directory for plots
    log_dir = os.path.dirname(csv_path)
    output_dir = os.path.join(log_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analyses
    analyze_g1_gains(df)
    analyze_c4_compensation(df)
    analyze_eigenvalues(df)
    analyze_a1_dynamics(df)
    analyze_m_matrix(df)
    analyze_control_effectiveness(df)
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    plot_g1_evolution(df, output_dir)
    plot_c4_and_eigenvalues(df, output_dir)
    plot_a1_vs_yaw_rate(df, output_dir)
    plot_gain_scaling_analysis(df, output_dir)
    plot_control_performance(df, output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Plots saved to: {output_dir}/")

if __name__ == "__main__":
    main()
