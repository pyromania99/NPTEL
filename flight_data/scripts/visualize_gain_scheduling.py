#!/usr/bin/env python3
"""
Visualize Gain Scheduling Behavior
Shows how attitude gains change with yaw rate
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys

def load_config(config_file):
    """Load control parameters"""
    with open(config_file, 'r') as f:
        return json.load(f)

def compute_scheduled_gains(yaw_rate, kp_base, kd_base, kp_min, kd_max, rate_threshold):
    """
    Compute scheduled gains at given yaw rate
    
    Scheduling formula (matching C++ code):
    schedule_factor = (tanh((|yaw_rate| - 5) * 4.1 / rate_threshold) + 1.0) / 2.0
    
    This provides:
    - Smooth transition using tanh
    - Centered around 5 rad/s
    - Scales with rate_threshold
    - Output range: [0, 1]
    """
    # Compute scheduling factor using tanh (matching C++ implementation)
    abs_rate = np.abs(yaw_rate)
    schedule_factor = (np.tanh((abs_rate - rate_threshold) / 3 * 4) + 1) / 2
    # (std::tanh((yaw_rate_mag - rate_threshold) / 3 * 4) + 1) / 2;
    # Interpolate gains
    # Decrease P gain with increasing yaw rate
    kp = kp_base + (kp_min - kp_base) * schedule_factor
    
    # Increase D gain with increasing yaw rate
    kd = kd_base + (kd_max - kd_base) * schedule_factor
    
    return kp, kd, schedule_factor

def plot_gain_scheduling(config_file):
    """Plot gain scheduling curves"""
    # Load config
    config = load_config(config_file)
    
    att = config['attitude']
    gs = config.get('gain_scheduling', {})
    
    kp_base = att['kp']
    kd_base = att['kd']
    kp_min = gs.get('kp_min', kp_base)
    kd_max = gs.get('kd_max', kd_base)
    rate_threshold = gs.get('rate_threshold', 5.0)
    enabled = gs.get('enabled', False)
    
    print("="*70)
    print("GAIN SCHEDULING CONFIGURATION")
    print("="*70)
    print(f"Enabled: {enabled}")
    print(f"\nBase Gains (at low rates):")
    print(f"  kp_base: {kp_base:.4f}")
    print(f"  kd_base: {kd_base:.4f}")
    print(f"\nScheduled Gains (at high rates):")
    print(f"  kp_min: {kp_min:.4f}")
    print(f"  kd_min: {kd_max:.4f}")
    print(f"\nScheduling:")
    print(f"  Rate threshold: {rate_threshold:.4f} rad/s")
    print(f"  P gain reduction: {((kp_base - kp_min)/kp_base * 100):.1f}%")
    print(f"  D gain increase: {((kd_max - kd_base)/kd_base * 100):.1f}%")
    print("="*70)
    
    # Generate yaw rate range
    yaw_rates = np.linspace(0, 10, 200)
    
    # Compute gains
    kp_scheduled = np.zeros_like(yaw_rates)
    kd_scheduled = np.zeros_like(yaw_rates)
    alpha_values = np.zeros_like(yaw_rates)
    
    for i, rate in enumerate(yaw_rates):
        kp, kd, alpha = compute_scheduled_gains(
            rate, kp_base, kd_base, kp_min, kd_max, rate_threshold
        )
        kp_scheduled[i] = kp
        kd_scheduled[i] = kd
        alpha_values[i] = alpha
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Attitude Controller Gain Scheduling', fontsize=16, fontweight='bold')
    
    # Plot 1: P gain
    ax = axes[0]
    ax.plot(yaw_rates, kp_scheduled, 'b-', linewidth=2, label='Scheduled kp')
    ax.axhline(kp_base, color='b', linestyle='--', alpha=0.5, label=f'kp_base = {kp_base:.3f}')
    ax.axhline(kp_min, color='r', linestyle='--', alpha=0.5, label=f'kp_min = {kp_min:.3f}')
    ax.axvline(rate_threshold, color='gray', linestyle=':', alpha=0.5, 
               label=f'Threshold = {rate_threshold:.1f} rad/s')
    ax.set_xlabel('Yaw Rate (rad/s)', fontsize=12)
    ax.set_ylabel('P Gain', fontsize=12)
    ax.set_title('Proportional Gain Scheduling (Decreases with Rate)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 10)
    
    # Plot 2: D gain
    ax = axes[1]
    ax.plot(yaw_rates, kd_scheduled, 'g-', linewidth=2, label='Scheduled kd')
    ax.axhline(kd_base, color='g', linestyle='--', alpha=0.5, label=f'kd_base = {kd_base:.3f}')
    ax.axhline(kd_max, color='r', linestyle='--', alpha=0.5, label=f'kd_max = {kd_max:.3f}')
    ax.axvline(rate_threshold, color='gray', linestyle=':', alpha=0.5,
               label=f'Threshold = {rate_threshold:.1f} rad/s')
    ax.set_xlabel('Yaw Rate (rad/s)', fontsize=12)
    ax.set_ylabel('D Gain', fontsize=12)
    ax.set_title('Derivative Gain Scheduling (Increases with Rate)', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 10)
    
    # Plot 3: Scheduling factor
    ax = axes[2]
    ax.plot(yaw_rates, alpha_values, 'purple', linewidth=2, label='Scheduling factor')
    ax.axvline(rate_threshold, color='gray', linestyle=':', alpha=0.5,
               label=f'Threshold = {rate_threshold:.1f} rad/s')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    ax.axvline(5.0, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, 
               label='Center = 5.0 rad/s')
    ax.fill_between(yaw_rates, 0, alpha_values, alpha=0.2, color='purple')
    ax.set_xlabel('Yaw Rate (rad/s)', fontsize=12)
    ax.set_ylabel('Scheduling Factor', fontsize=12)
    ax.set_title('Scheduling Factor: (tanh((|ω| - 5) × 4.1 / threshold) + 1) / 2', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    # Save plot
    output_file = '/home/pyro/ws_offboard_control/flight_data/outputs/gain_scheduling_plot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    plt.show()

def main():
    config_file = "/home/pyro/ws_offboard_control/flight_data/config/control_params.json"
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    plot_gain_scheduling(config_file)

if __name__ == "__main__":
    main()
