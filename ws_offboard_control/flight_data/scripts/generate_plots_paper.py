#!/usr/bin/env python3
"""
Plot generator for offboard_paper controller logs.
Visualizes state estimation, control outputs, and errors from equation (14) controller.
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

def load_config():
    """Load controller parameters from JSON config."""
    config_path = "/home/pyro/ws_offboard_control/flight_data/config/offboard_paper_params.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

def find_latest_log(source_name="offboard_control_pd"):
    """Find the most recent log file."""
    log_dir = f"/home/pyro/ws_offboard_control/flight_data/logs/{source_name}"
    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        sys.exit(1)
    
    log_files = glob.glob(os.path.join(log_dir, "flight_log_*.csv"))
    if not log_files:
        print(f"No log files found in {log_dir}")
        sys.exit(1)
    
    latest = max(log_files, key=os.path.getmtime)
    print(f"Loading: {latest}")
    return latest

def main():
    parser = argparse.ArgumentParser(description='Generate plots for offboard_paper controller')
    parser.add_argument('--source', type=str, default='offboard_control_pd', help='Source name for log directory')
    parser.add_argument('--file', type=str, default=None, help='Specific log file to plot')
    args = parser.parse_args()

    # Load data
    if args.file:
        log_file = args.file
    else:
        log_file = find_latest_log(args.source)
    
    df = pd.read_csv(log_file)
    t = df['timestamp']
    
    # Load config for reference
    config = load_config()
    
    # Controller gains (equation 14)
    k_n3 = config.get('controller_gains', {}).get('k_n3', 5.0)
    k_vz = config.get('controller_gains', {}).get('k_vz', 1.0)
    
    # Altitude PD controller
    k_pz = config.get('altitude_pd', {}).get('k_pz', 1.0)
    k_dz = config.get('altitude_pd', {}).get('k_dz', 0.5)
    z_d = config.get('altitude_pd', {}).get('z_d', -5.0)
    
    # Desired states
    x1_d_0 = config.get('desired_states', {}).get('x1_d_0', 0.0)
    x1_d_1 = config.get('desired_states', {}).get('x1_d_1', 0.0)
    
    # Physical parameters
    mass = config.get('physical_parameters', {}).get('mass', 2.0)
    g = config.get('physical_parameters', {}).get('g', 9.81)
    Jx = config.get('physical_parameters', {}).get('Jx', 0.02166666667)
    Jy = config.get('physical_parameters', {}).get('Jy', 0.02166666667)
    Jz = config.get('physical_parameters', {}).get('Jz', 0.04)
    
    # Run settings
    run_duration = config.get('run_settings', {}).get('run_duration_sec', 15.0)
    
    # Print config summary
    print("=== CONFIG SUMMARY ===")
    print(f"Controller: k_n3={k_n3}, k_vz={k_vz}")
    print(f"Altitude PD: k_pz={k_pz}, k_dz={k_dz}, z_d={z_d}m")
    print(f"Desired x1: [{x1_d_0}, {x1_d_1}]")
    print(f"Physical: mass={mass}kg, Jx={Jx}, Jy={Jy}, Jz={Jz}")
    print(f"Run duration: {run_duration}s")
    print("=" * 22)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    config_str = f"k_n3={k_n3}, k_vz={k_vz} | Alt PD: k_pz={k_pz}, k_dz={k_dz}, z_d={z_d}m"
    fig.suptitle(f'Offboard Paper Controller Analysis\n{os.path.basename(log_file)}\n{config_str}', fontsize=12)
    
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.25)
    
    # ============================================================
    # Row 1: Position and Velocity
    # ============================================================
    
    # 3D Position
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, df['pos_x'], 'r-', label='X', linewidth=1.5)
    ax1.plot(t, df['pos_y'], 'g-', label='Y', linewidth=1.5)
    ax1.plot(t, df['pos_z'], 'b-', label='Z', linewidth=1.5)
    ax1.axhline(y=z_d, color='b', linestyle='--', alpha=0.5, label=f'Z_des={z_d}m')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Position (NED)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Velocity
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, df['vel_x'], 'r-', label='Vx', linewidth=1.5)
    ax2.plot(t, df['vel_y'], 'g-', label='Vy', linewidth=1.5)
    ax2.plot(t, df['vel_z'], 'b-', label='Vz', linewidth=1.5)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Altitude tracking
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, -df['pos_z'], 'b-', label='Altitude', linewidth=1.5)
    ax3.axhline(y=-z_d, color='r', linestyle='--', label=f'Desired ({-z_d}m)', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Altitude (m, + up)')
    ax3.set_title('Altitude Tracking')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # ============================================================
    # Row 2: Attitude (n3 and Euler)
    # ============================================================
    
    # n3 vector (body z-axis in world frame)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, df['n3_x'], 'r-', label='n3_x', linewidth=1.5)
    ax4.plot(t, df['n3_y'], 'g-', label='n3_y', linewidth=1.5)
    ax4.plot(t, df['n3_z'], 'b-', label='n3_z', linewidth=1.5)
    # Desired x1 = [n3_x_d, n3_y_d] from config
    ax4.axhline(y=x1_d_0, color='r', linestyle='--', alpha=0.5, label=f'n3_x_d={x1_d_0}')
    ax4.axhline(y=x1_d_1, color='g', linestyle='--', alpha=0.5, label=f'n3_y_d={x1_d_1}')
    ax4.axhline(y=1, color='b', linestyle=':', alpha=0.3, label='n3_z_des=1')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('n3 components')
    ax4.set_title('Body Z-Axis (n3) in World Frame')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    
    # Euler angles
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, np.rad2deg(df['roll']), 'r-', label='Roll', linewidth=1.5)
    ax5.plot(t, np.rad2deg(df['pitch']), 'g-', label='Pitch', linewidth=1.5)
    ax5.plot(t, np.rad2deg(df['yaw']), 'b-', label='Yaw', linewidth=1.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angle (deg)')
    ax5.set_title('Euler Angles')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # Angular rates
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, np.rad2deg(df['p']), 'r-', label='p (roll rate)', linewidth=1.5)
    ax6.plot(t, np.rad2deg(df['q']), 'g-', label='q (pitch rate)', linewidth=1.5)
    ax6.plot(t, np.rad2deg(df['r']), 'b-', label='r (yaw rate)', linewidth=1.5)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Angular Rate (deg/s)')
    ax6.set_title('Body Angular Rates')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    
    # ============================================================
    # Row 3: Control Errors (x1_tilde, x2_tilde)
    # ============================================================
    
    # x1 error (attitude error)
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t, df['x1_tilde_0'], 'r-', label='x̃1_0 (n3_x err)', linewidth=1.5)
    ax7.plot(t, df['x1_tilde_1'], 'g-', label='x̃1_1 (n3_y err)', linewidth=1.5)
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Error')
    ax7.set_title('Attitude Error x̃1 = x1 - x1_d')
    ax7.legend(loc='best')
    ax7.grid(True, alpha=0.3)
    
    # x2 error (state error)
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t, df['x2_tilde_0'], 'r-', label='x̃2_0 (vz err)', linewidth=1.5)
    ax8.plot(t, df['x2_tilde_1'], 'g-', label='x̃2_1 (p err)', linewidth=1.5)
    ax8.plot(t, df['x2_tilde_2'], 'b-', label='x̃2_2 (q err)', linewidth=1.5)
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Error')
    ax8.set_title('State Error x̃2 = x2 - x2_d')
    ax8.legend(loc='best')
    ax8.grid(True, alpha=0.3)
    
    # State derivatives
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(t, df['x1_dot_0'], 'r-', label='ẋ1_0 (ṅ3_x)', linewidth=1.5)
    ax9.plot(t, df['x1_dot_1'], 'g-', label='ẋ1_1 (ṅ3_y)', linewidth=1.5)
    ax9.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Derivative')
    ax9.set_title('State Derivatives ẋ1')
    ax9.legend(loc='best')
    ax9.grid(True, alpha=0.3)
    
    # ============================================================
    # Row 4: Control Outputs
    # ============================================================
    
    # Desired control (u_d from eq 14)
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(t, df['u_d_thrust'], 'b-', label='u_d thrust (N)', linewidth=1.5)
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('Thrust (N)')
    ax10.set_title('Desired Thrust (Eq. 14)')
    ax10.legend(loc='best')
    ax10.grid(True, alpha=0.3)
    
    ax10b = ax10.twinx()
    ax10b.plot(t, df['u_d_tau_x'], 'r--', label='τ_x', linewidth=1)
    ax10b.plot(t, df['u_d_tau_y'], 'g--', label='τ_y', linewidth=1)
    ax10b.set_ylabel('Torque (Nm)', color='gray')
    ax10b.legend(loc='upper right')
    
    # Normalized thrust/torque setpoints
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.plot(t, df['thrust_z'], 'b-', label='Thrust Z (norm)', linewidth=1.5)
    ax11.plot(t, df['torque_x'], 'r-', label='Torque X (norm)', linewidth=1.5)
    ax11.plot(t, df['torque_y'], 'g-', label='Torque Y (norm)', linewidth=1.5)
    ax11.plot(t, df['torque_z'], 'm-', label='Torque Z (norm)', linewidth=1.5)
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('Normalized [-1, 1]')
    ax11.set_title('Published Thrust/Torque Setpoints')
    ax11.legend(loc='best')
    ax11.grid(True, alpha=0.3)
    
    # Motor outputs
    ax12 = fig.add_subplot(gs[3, 2])
    if 'motor1' in df.columns:
        ax12.plot(t, df['motor1'], 'r-', label='Motor 1', linewidth=1)
        ax12.plot(t, df['motor2'], 'g-', label='Motor 2', linewidth=1)
        ax12.plot(t, df['motor3'], 'b-', label='Motor 3', linewidth=1)
        ax12.plot(t, df['motor4'], 'm-', label='Motor 4', linewidth=1)
    ax12.set_xlabel('Time (s)')
    ax12.set_ylabel('Motor Command')
    ax12.set_title('Motor Outputs')
    ax12.legend(loc='best')
    ax12.grid(True, alpha=0.3)
    
    # Save figure
    plot_dir = "/home/pyro/ws_offboard_control/flight_data/plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(log_file))[0]
    plot_path = os.path.join(plot_dir, f"{base_name}_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
