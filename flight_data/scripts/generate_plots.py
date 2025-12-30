#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import json
import textwrap
import argparse

import glob
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate flight data analysis plots')
parser.add_argument('--source', type=str, default=None,
                    help='Source executable name (e.g., offboard_control_pd). Used to organize logs and outputs.')
parser.add_argument('--output-dir', type=str, default=None,
                    help='Override output directory for plots')
args = parser.parse_args()

# Determine log directory based on source
base_logs_dir = '/home/pyro/ws_offboard_control/flight_data/logs'
if args.source:
    logs_dir = os.path.join(base_logs_dir, args.source)
else:
    logs_dir = base_logs_dir

# Read data - find the most recent flight log
log_files = glob.glob(os.path.join(logs_dir, 'flight_log_*.csv'))
if not log_files:
    # Fallback: try base logs directory
    log_files = glob.glob(os.path.join(base_logs_dir, 'flight_log_*.csv'))
if not log_files:
    print(f'No flight log files found in {logs_dir}!')
    exit(1)
latest_log = max(log_files, key=os.path.getctime)
print(f'Reading log file: {latest_log}')
if args.source:
    print(f'Source executable: {args.source}')
df = pd.read_csv(latest_log)
t = df['timestamp']

# Try to load configuration snapshot next to CSV, else fall back to default config
cfg = None
cfg_source = None
snapshot_candidate = os.path.splitext(latest_log)[0] + '_config.json'
default_cfg = '/home/pyro/ws_offboard_control/flight_data/config/control_params.json'
for candidate in [snapshot_candidate, default_cfg]:
    if os.path.exists(candidate):
        try:
            with open(candidate, 'r') as f:
                cfg = json.load(f)
            cfg_source = candidate
            break
        except Exception as e:
            print(f"[plots] Failed to load configuration from {candidate}: {e}")
            cfg = None

def _get(cfg_obj, path, default=None):
    try:
        d = cfg_obj
        for p in path:
            d = d[p]
        return d
    except Exception:
        return default

def _format_config_summary(cfg_obj):
    if cfg_obj is None:
        return "No configuration file found."
    lines = []
    lines.append("=== CONTROLLER CONFIG SUMMARY ===")
    lines.append(
        f"Attitude: kp={_get(cfg_obj,['attitude','kp'], _get(cfg_obj,['attitude','kp_base'],4.0))}, "
        f"kd={_get(cfg_obj,['attitude','kd'], _get(cfg_obj,['attitude','kd_base'],0.0))}; "
        f"GainSched: enabled={_get(cfg_obj,['gain_scheduling','enabled'], False)}, "
        f"kp_min={_get(cfg_obj,['gain_scheduling','kp_min'],1.5)}, kd_max={_get(cfg_obj,['gain_scheduling','kd_max'],0.6)}, "
        f"rate_thresh={_get(cfg_obj,['gain_scheduling','rate_threshold'],5.0)}"
    )
    lines.append(
        f"Rate (NDI): kp={_get(cfg_obj,['ndi_rate','kp'],0.1)}, kd={_get(cfg_obj,['ndi_rate','kd'],0.0)}; "
        f"Feedback: kp={_get(cfg_obj,['feedback_rate','kp'],0.18)}, kd={_get(cfg_obj,['feedback_rate','kd'],0.01)}"
    )
    lines.append(
        f"INDI: enabled={_get(cfg_obj,['indi','enabled'],False)}, scale={_get(cfg_obj,['indi','scale'],0.75)}, "
        f"blend={_get(cfg_obj,['indi','blend_ratio'],0.3)}, alpha_omega={_get(cfg_obj,['indi','alpha_omega_lp'],0.15)}, "
        f"alpha_acc={_get(cfg_obj,['indi','alpha_acc_lp'],0.25)}"
    )
    lines.append(
        f"Pos XY: kp=[{_get(cfg_obj,['position_xy','kp_x'],0.55)},{_get(cfg_obj,['position_xy','kp_y'],0.55)}], "
        f"kd=[{_get(cfg_obj,['position_xy','kd_x'],1.10)},{_get(cfg_obj,['position_xy','kd_y'],1.10)}], "
        f"ki=[{_get(cfg_obj,['position_xy','ki_x'],0.03)},{_get(cfg_obj,['position_xy','ki_y'],0.03)}], "
        f"vel_lpf={_get(cfg_obj,['position_xy','vel_lpf_alpha'],None)}, max_xy_acc_step={_get(cfg_obj,['position_xy','max_xy_acc_step'],None)}"
    )
    lines.append(
        f"Pos Z: kp={_get(cfg_obj,['position_z','kp'],20.0)}, kd={_get(cfg_obj,['position_z','kd'],20.0)}, ki={_get(cfg_obj,['position_z','ki'],1.0)}"
    )
    lines.append(
        f"Limits: max_horz_acc={_get(cfg_obj,['limits','max_horz_acc'],3.0)}, max_vert_acc={_get(cfg_obj,['limits','max_vert_acc'],None)}, max_tilt={_get(cfg_obj,['limits','max_tilt_angle'],0.7)}"
    )
    lines.append(
        f"Inertia: Ix={_get(cfg_obj,['inertia','Ix'],None)}, Iy={_get(cfg_obj,['inertia','Iy'],None)}, Iz={_get(cfg_obj,['inertia','Iz'],None)}"
    )
    lines.append(
        f"Yaw FF: dt_base={_get(cfg_obj,['yaw_ff','dt_base'],0.035)}, dt_gain={_get(cfg_obj,['yaw_ff','dt_gain'],0.008)}, "
        f"dt_max={_get(cfg_obj,['yaw_ff','dt_max'],0.1)}, lpf_alpha={_get(cfg_obj,['yaw_ff','lpf_alpha'],0.2)}"
    )
    pos_design = _get(cfg_obj,['targets','pos_design'],None)
    yaw_target = _get(cfg_obj,['targets','yaw_target'],None)
    if pos_design is not None or yaw_target is not None:
        lines.append(f"Targets: pos_design={pos_design}, yaw_target={yaw_target}")
    return "\n".join(lines)

config_summary = _format_config_summary(cfg)
if cfg_source:
    print(f"\nUsing configuration from: {cfg_source}")
print(config_summary)

# Create comprehensive figure with subplots
plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(40, 65))  # Increased height to accommodate all plots including new torque allocation plots
gs = GridSpec(21, 3, figure=fig, hspace=0.3, wspace=0.3)  # Added 2 more rows for torque allocation plots

# Create mask for data after 10 seconds (for better y-limit calculation)
steady_state_mask = t >= 10.0

# 1. Position tracking
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, df['pos_x'], 'b-', label='Actual X', linewidth=2)
ax1.plot(t, df['pos_des_x'], 'b--', label='Desired X', linewidth=2)
ax1.plot(t, df['pos_y'], 'r-', label='Actual Y', linewidth=2)
ax1.plot(t, df['pos_des_y'], 'r--', label='Desired Y', linewidth=2)
ax1.plot(t, df['pos_z'], 'g-', label='Actual Z', linewidth=2)
ax1.plot(t, df['pos_des_z'], 'g--', label='Desired Z', linewidth=2)
# Calculate y-limits from steady-state data only
if steady_state_mask.any():
    pos_data_ss = pd.concat([df.loc[steady_state_mask, 'pos_x'], df.loc[steady_state_mask, 'pos_y'], 
                             df.loc[steady_state_mask, 'pos_z'], df.loc[steady_state_mask, 'pos_des_x'],
                             df.loc[steady_state_mask, 'pos_des_y'], df.loc[steady_state_mask, 'pos_des_z']])
    pos_min, pos_max = pos_data_ss.min(), pos_data_ss.max()
    pos_margin = (pos_max - pos_min) * 0.1
    ax1.set_ylim(pos_min - pos_margin, pos_max + pos_margin)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Position (m)')
ax1.set_title('Position Tracking Performance')
ax1.legend()
ax1.grid(True)

# 2. Velocity tracking
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, df['vel_x'], 'b-', label='Raw X', linewidth=1.5)
ax2.plot(t, df['vel_filt_x'], 'b--', label='Filtered X', linewidth=2)
ax2.plot(t, df['vel_y'], 'r-', label='Raw Y', linewidth=1.5)
ax2.plot(t, df['vel_filt_y'], 'r--', label='Filtered Y', linewidth=2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('XY Velocity Filtering')
ax2.legend()
ax2.grid(True)

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t, df['vel_z'], 'g-', label='Z Velocity', linewidth=2)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Z Velocity (m/s)')
ax3.set_title('Vertical Velocity')
ax3.legend()
ax3.grid(True)

# 3. Attitude tracking - Yaw (full row)
ax4b = fig.add_subplot(gs[2, :])
ax4b.plot(t, np.degrees(df['yaw']), 'g-', label='Yaw', linewidth=2)
ax4b.plot(t, np.degrees(df['yaw_des']), 'g--', label='Yaw Desired', linewidth=2)
ax4b.set_xlabel('Time (s)')
ax4b.set_ylabel('Yaw (deg)')
ax4b.set_title('Yaw Tracking')
ax4b.legend()
ax4b.grid(True)

# 3b. Attitude tracking - Roll & Pitch
ax4 = fig.add_subplot(gs[3, 0:3])
ax4.plot(t, np.degrees(df['roll']), 'b-', label='Roll', linewidth=2)
ax4.plot(t, np.degrees(df['roll_des']), 'b--', label='Roll Desired', linewidth=2)
ax4.plot(t, np.degrees(df['pitch']), 'r-', label='Pitch', linewidth=2)
ax4.plot(t, np.degrees(df['pitch_des']), 'r--', label='Pitch Desired', linewidth=2)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Attitude (deg)')
ax4.set_title('Roll & Pitch Tracking')
ax4.legend()
ax4.grid(True)

# 4. Angular rates
ax5 = fig.add_subplot(gs[4, 0])
ax5.plot(t, np.degrees(df['p']), 'b-', label='Actual p', linewidth=2)
ax5.plot(t, np.degrees(df['p_des']), 'b--', label='Desired p', linewidth=2)
# Calculate y-limits from steady-state data
if steady_state_mask.any():
    p_data_ss = pd.concat([np.degrees(df.loc[steady_state_mask, 'p']), 
                           np.degrees(df.loc[steady_state_mask, 'p_des'])])
    p_min, p_max = p_data_ss.min(), p_data_ss.max()
    p_margin = (p_max - p_min) * 0.1
    ax5.set_ylim(p_min - p_margin, p_max + p_margin)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Roll Rate (deg/s)')
ax5.set_title('Roll Rate Tracking')
ax5.legend()
ax5.grid(True)

ax6 = fig.add_subplot(gs[4, 1])
ax6.plot(t, np.degrees(df['q']), 'r-', label='Actual q', linewidth=2)
ax6.plot(t, np.degrees(df['q_des']), 'r--', label='Desired q', linewidth=2)
# Calculate y-limits from steady-state data
if steady_state_mask.any():
    q_data_ss = pd.concat([np.degrees(df.loc[steady_state_mask, 'q']), 
                           np.degrees(df.loc[steady_state_mask, 'q_des'])])
    q_min, q_max = q_data_ss.min(), q_data_ss.max()
    q_margin = (q_max - q_min) * 0.1
    ax6.set_ylim(q_min - q_margin, q_max + q_margin)
ax6.set_xlabel('Time (s)')
ax6.set_ylabel('Pitch Rate (deg/s)')
ax6.set_title('Pitch Rate Tracking')
ax6.legend()
ax6.grid(True)

ax7 = fig.add_subplot(gs[4, 2])
ax7.plot(t, np.degrees(df['r']), 'g-', label='Actual r', linewidth=2)
ax7.plot(t, np.degrees(df['r_des']), 'g--', label='Desired r', linewidth=2)
ax7.set_xlabel('Time (s)')
ax7.set_ylabel('Yaw Rate (deg/s)')
ax7.set_title('Yaw Rate Tracking')
ax7.legend()
ax7.grid(True)

# 4b. Euler rates (phi_dot, theta_dot, psi_dot) - compute from body rates using T matrix
# T(phi, theta) transforms body rates to Euler rates
def compute_euler_rates(roll, pitch, p, q, r):
    """Convert body rates to Euler rates using T matrix"""
    phi_dot = p + np.sin(roll) * np.tan(pitch) * q + np.cos(roll) * np.tan(pitch) * r
    theta_dot = np.cos(roll) * q - np.sin(roll) * r
    psi_dot = (np.sin(roll) / np.cos(pitch)) * q + (np.cos(roll) / np.cos(pitch)) * r
    return phi_dot, theta_dot, psi_dot

# Compute Euler rates from logged body rates and attitudes
phi_dot, theta_dot, psi_dot = compute_euler_rates(
    df['roll'].values, df['pitch'].values,
    df['p'].values, df['q'].values, df['r'].values
)

# Desired Euler rates (from p_des, q_des, r_des if they represent Euler rates, or compute from body)
phi_dot_des, theta_dot_des, psi_dot_des = compute_euler_rates(
    df['roll_des'].values, df['pitch_des'].values,
    df['p_des'].values, df['q_des'].values, df['r_des'].values
)

ax8 = fig.add_subplot(gs[5, 0])
ax8.plot(t, np.degrees(phi_dot), 'b-', label='Actual φ̇', linewidth=2)
ax8.plot(t, np.degrees(phi_dot_des), 'b--', label='Desired φ̇', linewidth=2)
if steady_state_mask.any():
    phi_dot_ss = np.concatenate([np.degrees(phi_dot[steady_state_mask]), 
                                  np.degrees(phi_dot_des[steady_state_mask])])
    phi_dot_min, phi_dot_max = phi_dot_ss.min(), phi_dot_ss.max()
    phi_dot_margin = (phi_dot_max - phi_dot_min) * 0.1
    ax8.set_ylim(phi_dot_min - phi_dot_margin, phi_dot_max + phi_dot_margin)
ax8.set_xlabel('Time (s)')
ax8.set_ylabel('Roll Euler Rate (deg/s)')
ax8.set_title('Roll Euler Rate (φ̇)')
ax8.legend()
ax8.grid(True)

ax9 = fig.add_subplot(gs[5, 1])
ax9.plot(t, np.degrees(theta_dot), 'r-', label='Actual θ̇', linewidth=2)
ax9.plot(t, np.degrees(theta_dot_des), 'r--', label='Desired θ̇', linewidth=2)
if steady_state_mask.any():
    theta_dot_ss = np.concatenate([np.degrees(theta_dot[steady_state_mask]), 
                                    np.degrees(theta_dot_des[steady_state_mask])])
    theta_dot_min, theta_dot_max = theta_dot_ss.min(), theta_dot_ss.max()
    theta_dot_margin = (theta_dot_max - theta_dot_min) * 0.1
    ax9.set_ylim(theta_dot_min - theta_dot_margin, theta_dot_max + theta_dot_margin)
ax9.set_xlabel('Time (s)')
ax9.set_ylabel('Pitch Euler Rate (deg/s)')
ax9.set_title('Pitch Euler Rate (θ̇)')
ax9.legend()
ax9.grid(True)

ax9b = fig.add_subplot(gs[5, 2])
ax9b.plot(t, np.degrees(psi_dot), 'g-', label='Actual ψ̇', linewidth=2)
ax9b.plot(t, np.degrees(psi_dot_des), 'g--', label='Desired ψ̇', linewidth=2)
ax9b.set_xlabel('Time (s)')
ax9b.set_ylabel('Yaw Euler Rate (deg/s)')
ax9b.set_title('Yaw Euler Rate (ψ̇)')
ax9b.legend()
ax9b.grid(True)

# 5. p Error (full row)
ax10 = fig.add_subplot(gs[6, :])
ax10_torque = ax10.twinx()
ax10.plot(t, np.degrees(df['rate_err_p']), 'b-', linewidth=2, label='Roll Rate Error')

# Stacked torque components for X axis with blend ratio weighting
if all(col in df.columns for col in ['tau_ndi_x', 'tau_indi_x', 'indi_blend_ratio']):
    # Apply blend ratio weighting: tau_cmd = (1-blend)*NDI + blend*INDI + feedback
    tau_ndi_weighted_x = df['tau_ndi_x'] * (1.0 - df['indi_blend_ratio'])
    tau_indi_weighted_x = df['tau_indi_x'] * df['indi_blend_ratio']
    tau_fb_x = df['torque_x'] - tau_ndi_weighted_x - tau_indi_weighted_x
    
    # Stacked area plot
    ax10_torque.fill_between(t, 0, tau_ndi_weighted_x, label='NDI', color='orange', alpha=0.4, linewidth=0)
    ax10_torque.fill_between(t, tau_ndi_weighted_x, tau_ndi_weighted_x + tau_indi_weighted_x, 
                             label='INDI', color='yellow', alpha=0.4, linewidth=0)
    ax10_torque.fill_between(t, tau_ndi_weighted_x + tau_indi_weighted_x, df['torque_x'], 
                             label='Feedback', color='cyan', alpha=0.4, linewidth=0)
    ax10_torque.plot(t, df['torque_x'], 'c-', linewidth=0.5, alpha=0.8, label='Total X Torque')
    print('[plots] Roll (X) torque: using blend-weighted stacked component visualization')
else:
    # Fallback to simple line
    ax10_torque.plot(t, df['torque_x'], 'c-', linewidth=1.5, alpha=0.7, label='X Torque')
    print('[plots] Roll (X) torque: component columns not found, using simple plot')

# Align zero lines - use steady-state data for limits
if steady_state_mask.any():
    err_max = max(abs(np.degrees(df.loc[steady_state_mask, 'rate_err_p']).min()), 
                  abs(np.degrees(df.loc[steady_state_mask, 'rate_err_p']).max()))
    torque_max = max(abs(df.loc[steady_state_mask, 'torque_x'].min()), 
                     abs(df.loc[steady_state_mask, 'torque_x'].max()))
else:
    err_max = max(abs(np.degrees(df['rate_err_p']).min()), abs(np.degrees(df['rate_err_p']).max()))
    torque_max = max(abs(df['torque_x'].min()), abs(df['torque_x'].max()))
ax10.set_ylim(-err_max * 1.1, err_max * 1.1)
ax10_torque.set_ylim(-torque_max * 1.1, torque_max * 1.1)

# ax10.set_xlabel('Time (s)')
ax10.set_ylabel('Roll Rate Error (deg/s)', color='b')
ax10_torque.set_ylabel('X Torque (N·m)', color='c')
ax10.set_title('Roll Rate (p) Error & X Torque Command')
ax10.tick_params(axis='y', labelcolor='b')
ax10_torque.tick_params(axis='y', labelcolor='c')
ax10.legend(loc='upper left')
ax10_torque.legend(loc='upper right')
ax10.grid(True)

# 6. q Error (full row)
ax11 = fig.add_subplot(gs[7, :])
ax11_torque = ax11.twinx()
ax11.plot(t, np.degrees(df['rate_err_q']), 'r-', linewidth=2, label='Pitch Rate Error')

# Stacked torque components for Y axis with blend ratio weighting
if all(col in df.columns for col in ['tau_ndi_y', 'tau_indi_y', 'indi_blend_ratio']):
    # Apply blend ratio weighting: tau_cmd = (1-blend)*NDI + blend*INDI + feedback
    tau_ndi_weighted_y = df['tau_ndi_y'] * (1.0 - df['indi_blend_ratio'])
    tau_indi_weighted_y = df['tau_indi_y'] * df['indi_blend_ratio']
    tau_fb_y = df['torque_y'] - tau_ndi_weighted_y - tau_indi_weighted_y
    
    # Stacked area plot
    ax11_torque.fill_between(t, 0, tau_ndi_weighted_y, label='NDI', color='purple', alpha=0.4, linewidth=0)
    ax11_torque.fill_between(t, tau_ndi_weighted_y, tau_ndi_weighted_y + tau_indi_weighted_y, 
                             label='INDI', color='orange', alpha=0.4, linewidth=0)
    ax11_torque.fill_between(t, tau_ndi_weighted_y + tau_indi_weighted_y, df['torque_y'], 
                             label='Feedback', color='pink', alpha=0.4, linewidth=0)
    ax11_torque.plot(t, df['torque_y'], 'm-', linewidth=1.5, alpha=0.8, label='Total Y Torque')
    print('[plots] Pitch (Y) torque: using blend-weighted stacked component visualization')
else:
    # Fallback to simple line
    ax11_torque.plot(t, df['torque_y'], 'm-', linewidth=0.5, alpha=0.7, label='Y Torque')
    print('[plots] Pitch (Y) torque: component columns not found, using simple plot')


if steady_state_mask.any():
    err_max = max(abs(np.degrees(df.loc[steady_state_mask, 'rate_err_q']).min()), 
                  abs(np.degrees(df.loc[steady_state_mask, 'rate_err_q']).max()))
    torque_max = max(abs(df.loc[steady_state_mask, 'torque_y'].min()), 
                     abs(df.loc[steady_state_mask, 'torque_y'].max()))
else:
    err_max = max(abs(np.degrees(df['rate_err_q']).min()), abs(np.degrees(df['rate_err_q']).max()))
    torque_max = max(abs(df['torque_y'].min()), abs(df['torque_y'].max()))
ax11.set_ylim(-err_max * 1.1, err_max * 1.1)
ax11_torque.set_ylim(-torque_max * 1.1, torque_max * 1.1)

# ax11.set_xlabel('Time (s)')
ax11.set_ylabel('Pitch Rate Error (deg/s)', color='r')
ax11_torque.set_ylabel('Y Torque (N·m)', color='m')
ax11.set_title('Pitch Rate (q) Error & Y Torque Command')
ax11.tick_params(axis='y', labelcolor='r')
ax11_torque.tick_params(axis='y', labelcolor='m')
ax11.legend(loc='upper left')
ax11_torque.legend(loc='upper right')
ax11.grid(True)

# 7. Attitude Error (full row)
ax9 = fig.add_subplot(gs[8, :])
ax9.plot(t, np.degrees(df['att_err_roll']), 'b-', label='Roll Error', linewidth=2)
ax9.plot(t, np.degrees(df['att_err_pitch']), 'r-', label='Pitch Error', linewidth=2)
# ax9.set_xlabel('Time (s)')
ax9.set_ylabel('Attitude Error (deg)')
ax9.set_title('Attitude Errors')
ax9.legend()
ax9.grid(True)

# NEW: Angular acceleration plots (p_dot, q_dot, r_dot) with torque stacks
# Check if angular acceleration columns exist
has_ang_accel = all(col in df.columns for col in ['p_dot_des', 'q_dot_des', 'r_dot_des',
                                                    'p_dot_meas', 'q_dot_meas', 'r_dot_meas'])

if has_ang_accel:
    # 8. p_dot (Roll angular acceleration) with X torque stack
    ax_pdot = fig.add_subplot(gs[9, :])
    ax_pdot_torque = ax_pdot.twinx()
    
    ax_pdot.plot(t, np.degrees(df['p_dot_des']), 'r--', linewidth=0.5, label='Desired p_dot', alpha=0.8)
    # ax_pdot.plot(t, np.degrees(df['p_dot_meas']), 'b-', linewidth=0.5, label='Measured p_dot')
    
    # # Stacked torque components for X axis
    if all(col in df.columns for col in ['tau_ndi_x', 'tau_indi_x', 'indi_blend_ratio']):
        tau_ndi_weighted_x = df['tau_ndi_x'] * (1.0 - df['indi_blend_ratio'])
        tau_indi_weighted_x = df['tau_indi_x'] * df['indi_blend_ratio']
        tau_fb_x = df['torque_x'] - tau_ndi_weighted_x - tau_indi_weighted_x

        ax_pdot_torque.fill_between(t, 0, tau_ndi_weighted_x, label='NDI', color='purple', alpha=0.4, linewidth=0)
    #     ax_pdot_torque.fill_between(t, tau_ndi_weighted_x, tau_ndi_weighted_x + tau_indi_weighted_x, 
    #                                  label='INDI', color='yellow', alpha=0.4, linewidth=0)
        ax_pdot_torque.fill_between(t, tau_ndi_weighted_x + tau_indi_weighted_x, df['torque_x'], 
                                     label='Feedback', color='orange', alpha=0.4, linewidth=0)
        # ax_pdot_torque.plot(t, df['torque_x'], 'c-', linewidth=0.1, alpha=0.8, label='Total X Torque')
    else:
        ax_pdot_torque.plot(t, df['torque_x'], 'c-', linewidth=0.1, alpha=0.7, label='X Torque')
    
    # Add coupling and net torque overlays if available
    if 'coupling_x' in df.columns:
        net_torque_x = df['torque_x'] - df['coupling_x']  # Motors need less when coupling assists
        ax_pdot_torque.plot(t, df['coupling_x'], 'k:', linewidth=1.5, alpha=0.6, label='Coupling X')
        ax_pdot_torque.plot(t, net_torque_x, 'c-', linewidth=0.7, alpha=0.9, label='Net Torque X')
    
    # Align axes (include net torque if available)
    if steady_state_mask.any():
        pdot_max = max(abs(np.degrees(df.loc[steady_state_mask, 'p_dot_des']).max()),
                       abs(np.degrees(df.loc[steady_state_mask, 'p_dot_des']).min()),
                       abs(np.degrees(df.loc[steady_state_mask, 'p_dot_meas']).max()),
                       abs(np.degrees(df.loc[steady_state_mask, 'p_dot_meas']).min()))
        if 'coupling_x' in df.columns:
            net_x = df['torque_x'] - df['coupling_x']  # Subtract coupling
            torque_max = max(abs(df.loc[steady_state_mask, 'torque_x'].min()), 
                           abs(df.loc[steady_state_mask, 'torque_x'].max()),
                           abs(net_x.loc[steady_state_mask].min()),
                           abs(net_x.loc[steady_state_mask].max()),
                           abs(df.loc[steady_state_mask, 'coupling_x'].min()),
                           abs(df.loc[steady_state_mask, 'coupling_x'].max()))
        else:
            torque_max = max(abs(df.loc[steady_state_mask, 'torque_x'].min()), 
                           abs(df.loc[steady_state_mask, 'torque_x'].max()))
    else:
        pdot_max = max(abs(np.degrees(df['p_dot_des']).max()), abs(np.degrees(df['p_dot_meas']).max()))
        if 'coupling_x' in df.columns:
            net_x = df['torque_x'] - df['coupling_x']
            torque_max = max(abs(df['torque_x'].min()), abs(df['torque_x'].max()),
                           abs(net_x.min()), abs(net_x.max()),
                           abs(df['coupling_x'].min()), abs(df['coupling_x'].max()))
        else:
            torque_max = max(abs(df['torque_x'].min()), abs(df['torque_x'].max()))
    
    ax_pdot.set_ylim(-pdot_max * 1.1, pdot_max * 1.1)
    ax_pdot_torque.set_ylim(-torque_max * 1.1, torque_max * 1.1)
    ax_pdot.set_ylabel('Roll Ang. Accel. (deg/s²)', color='b')
    ax_pdot_torque.set_ylabel('X Torque (N·m)', color='c')
    ax_pdot.set_title('Roll Angular Acceleration (p_dot) & X Torque Command')
    ax_pdot.tick_params(axis='y', labelcolor='b')
    ax_pdot_torque.tick_params(axis='y', labelcolor='c')
    ax_pdot.legend(loc='upper left')
    ax_pdot_torque.legend(loc='upper right')
    ax_pdot.grid(True)
    
    # 9. q_dot (Pitch angular acceleration) with Y torque stack
    ax_qdot = fig.add_subplot(gs[10, :])
    ax_qdot_torque = ax_qdot.twinx()
    
    ax_qdot.plot(t, np.degrees(df['q_dot_des']), 'b--', linewidth=0.5, label='Desired q_dot', alpha=0.8)
    ax_qdot.plot(t, np.degrees(df['q_dot_meas']), 'r-', linewidth=0.5, label='Measured q_dot')
    ax_qdot_torque.plot(t, df['torque_y'], 'm-', linewidth=0.1, alpha=0.7, label='Y Torque')
    
    if 'coupling_y' in df.columns:
        net_torque_y = df['torque_y'] - df['coupling_y']  # Motors need less when coupling assists
        ax_qdot_torque.plot(t, df['coupling_y'], 'k:', linewidth=1.5, alpha=0.6, label='Coupling Y')
        ax_qdot_torque.plot(t, net_torque_y, 'm-', linewidth=0.7, alpha=0.9, label='Net Torque Y')
    
    # Align axes (include net torque if available)
    if steady_state_mask.any():
        qdot_max = max(abs(np.degrees(df.loc[steady_state_mask, 'q_dot_des']).max()),
                       abs(np.degrees(df.loc[steady_state_mask, 'q_dot_des']).min()),
                       abs(np.degrees(df.loc[steady_state_mask, 'q_dot_meas']).max()),
                       abs(np.degrees(df.loc[steady_state_mask, 'q_dot_meas']).min()))
        if 'coupling_y' in df.columns:
            net_y = df['torque_y'] - df['coupling_y']  # Subtract coupling
            torque_max = max(abs(df.loc[steady_state_mask, 'torque_y'].min()), 
                           abs(df.loc[steady_state_mask, 'torque_y'].max()),
                           abs(net_y.loc[steady_state_mask].min()),
                           abs(net_y.loc[steady_state_mask].max()),
                           abs(df.loc[steady_state_mask, 'coupling_y'].min()),
                           abs(df.loc[steady_state_mask, 'coupling_y'].max()))
        else:
            torque_max = max(abs(df.loc[steady_state_mask, 'torque_y'].min()), 
                           abs(df.loc[steady_state_mask, 'torque_y'].max()))
    else:
        qdot_max = max(abs(np.degrees(df['q_dot_des']).max()), abs(np.degrees(df['q_dot_meas']).max()))
        if 'coupling_y' in df.columns:
            net_y = df['torque_y'] + df['coupling_y']
            torque_max = max(abs(df['torque_y'].min()), abs(df['torque_y'].max()),
                           abs(net_y.min()), abs(net_y.max()),
                           abs(df['coupling_y'].min()), abs(df['coupling_y'].max()))
        else:
            torque_max = max(abs(df['torque_y'].min()), abs(df['torque_y'].max()))
    
    ax_qdot.set_ylim(-qdot_max * 1.1, qdot_max * 1.1)
    ax_qdot_torque.set_ylim(-torque_max * 1.1, torque_max * 1.1)
    ax_qdot.set_ylabel('Pitch Ang. Accel. (deg/s²)', color='r')
    ax_qdot_torque.set_ylabel('Y Torque (N·m)', color='m')
    ax_qdot.set_title('Pitch Angular Acceleration (q_dot) & Y Torque Command')
    ax_qdot.tick_params(axis='y', labelcolor='r')
    ax_qdot_torque.tick_params(axis='y', labelcolor='m')
    ax_qdot.legend(loc='upper left')
    ax_qdot_torque.legend(loc='upper right')
    ax_qdot.grid(True)
    
    # 10. r_dot (Yaw angular acceleration) with Z torque stack
    # ax_rdot = fig.add_subplot(gs[9, :])
    # ax_rdot_torque = ax_rdot.twinx()
    
    # # ax_rdot.plot(t, np.degrees(df['r_dot_des']), 'r--', linewidth=0.5, label='Desired r_dot', alpha=0.8)
    # # ax_rdot.plot(t, np.degrees(df['r_dot_meas']), 'g-', linewidth=0.5, label='Measured r_dot')
    
    # # Stacked torque components for Z axis
    # if all(col in df.columns for col in ['tau_ndi_z', 'tau_indi_z', 'indi_blend_ratio']):
    #     tau_ndi_weighted_z = df['tau_ndi_z'] * (1.0 - df['indi_blend_ratio'])
    #     tau_indi_weighted_z = df['tau_indi_z'] * df['indi_blend_ratio']
    #     tau_fb_z = df['torque_z'] - tau_ndi_weighted_z - tau_indi_weighted_z
        
    #     ax_rdot_torque.fill_between(t, 0, tau_ndi_weighted_z, label='NDI', color='purple', alpha=0.4, linewidth=0)
    #     ax_rdot_torque.fill_between(t, tau_ndi_weighted_z, tau_ndi_weighted_z + tau_indi_weighted_z, 
    #                                  label='INDI', color='yellow', alpha=0.4, linewidth=0)
    #     ax_rdot_torque.fill_between(t, tau_ndi_weighted_z + tau_indi_weighted_z, df['torque_z'], 
    #                                  label='Feedback', color='lightgreen', alpha=0.4, linewidth=0)
    #     # ax_rdot_torque.plot(t, df['torque_z'], 'purple', linewidth=0.01, alpha=0.8, label='Total Z Torque')
    # else:
    #     ax_rdot_torque.plot(t, df['torque_z'], 'g-', linewidth=0.1, alpha=0.7, label='Z Torque')
    
    # # Align axes
    # if steady_state_mask.any():
    #     rdot_max = max(abs(np.degrees(df.loc[steady_state_mask, 'r_dot_des']).max()),
    #                    abs(np.degrees(df.loc[steady_state_mask, 'r_dot_des']).min()),
    #                    abs(np.degrees(df.loc[steady_state_mask, 'r_dot_meas']).max()),
    #                    abs(np.degrees(df.loc[steady_state_mask, 'r_dot_meas']).min()))
    #     torque_max = max(abs(df.loc[steady_state_mask, 'torque_z'].min()), 
    #                      abs(df.loc[steady_state_mask, 'torque_z'].max()))
    # else:
    #     rdot_max = max(abs(np.degrees(df['r_dot_des']).max()), abs(np.degrees(df['r_dot_meas']).max()))
    #     torque_max = max(abs(df['torque_z'].min()), abs(df['torque_z'].max()))
    
    # ax_rdot.set_ylim(-rdot_max * 1.1, rdot_max * 1.1)
    # ax_rdot_torque.set_ylim(-torque_max * 1.1, torque_max * 1.1)
    # ax_rdot.set_ylabel('Yaw Ang. Accel. (deg/s²)', color='g')
    # ax_rdot_torque.set_ylabel('Z Torque (N·m)', color='darkgreen')
    # ax_rdot.set_title('Yaw Angular Acceleration (r_dot) & Z Torque Command')
    # ax_rdot.tick_params(axis='y', labelcolor='g')
    # ax_rdot_torque.tick_params(axis='y', labelcolor='darkgreen')
    # ax_rdot.legend(loc='upper left')
    # ax_rdot_torque.legend(loc='upper right')
    # ax_rdot.grid(True)
    
    print('[plots] Angular acceleration plots created successfully')
    
    # Shift remaining plots down by 5 rows: gs[0-5] taken (pos, vel, att-rp, yaw, rates, euler rates), gs[6-8] rate errors + att error, gs[9-10] ang accel
    row_offset = 5
else:
    print('[plots] Warning: Angular acceleration columns not found, skipping ang accel plots')
    row_offset = 3  # Shift by 3 for yaw plot + attitude error plot + euler rates

# NEW: Coupling terms and net torque plot
has_coupling = all(col in df.columns for col in ['coupling_x', 'coupling_y', 'coupling_z'])

if has_coupling:
    # Row 9+offset: Coupling terms with net torque
    ax_coupling = fig.add_subplot(gs[6 + row_offset, :])
    ax_net = ax_coupling.twinx()
    
    # Plot coupling terms on left axis
    ax_coupling.plot(t, df['coupling_x'], 'b-', linewidth=1.5, alpha=0.6, label='Coupling X')
    ax_coupling.plot(t, df['coupling_y'], 'r-', linewidth=1.5, alpha=0.6, label='Coupling Y')
    ax_coupling.plot(t, df['coupling_z'], 'g-', linewidth=1.5, alpha=0.6, label='Coupling Z')
    
    # Calculate and plot net torque on right axis
    net_torque_x = df['torque_x'] - df['coupling_x']  # What motors actually produce
    net_torque_y = df['torque_y'] - df['coupling_y']
    net_torque_z = df['torque_z'] - df['coupling_z']
    
    ax_net.plot(t, net_torque_x, 'b--', linewidth=2, alpha=0.8, label='Net X')
    ax_net.plot(t, net_torque_y, 'r--', linewidth=2, alpha=0.8, label='Net Y')
    ax_net.plot(t, net_torque_z, 'g--', linewidth=2, alpha=0.8, label='Net Z')
    
    # Align axes to have zero at same point
    if steady_state_mask.any():
        coupling_max = max(abs(df.loc[steady_state_mask, ['coupling_x', 'coupling_y', 'coupling_z']].min().min()),
                          abs(df.loc[steady_state_mask, ['coupling_x', 'coupling_y', 'coupling_z']].max().max()))
        net_max = max(abs(pd.concat([net_torque_x[steady_state_mask], 
                                     net_torque_y[steady_state_mask], 
                                     net_torque_z[steady_state_mask]]).min()),
                     abs(pd.concat([net_torque_x[steady_state_mask], 
                                   net_torque_y[steady_state_mask], 
                                   net_torque_z[steady_state_mask]]).max()))
    else:
        coupling_max = max(abs(df[['coupling_x', 'coupling_y', 'coupling_z']].min().min()),
                          abs(df[['coupling_x', 'coupling_y', 'coupling_z']].max().max()))
        net_max = max(abs(pd.concat([net_torque_x, net_torque_y, net_torque_z]).min()),
                     abs(pd.concat([net_torque_x, net_torque_y, net_torque_z]).max()))
    
    ax_coupling.set_ylim(-coupling_max * 1.1, coupling_max * 1.1)
    ax_net.set_ylim(-net_max * 1.1, net_max * 1.1)
    
    ax_coupling.set_ylabel('Coupling Torque ω×(I·ω) [N·m]', color='black')
    ax_net.set_ylabel('Net Torque (Cmd + Coupling) [N·m]', color='black')
    ax_coupling.set_title('Gyroscopic Coupling (Solid) and Net Torque (Dashed)')
    ax_coupling.tick_params(axis='y')
    ax_net.tick_params(axis='y')
    ax_coupling.legend(loc='upper left', fontsize=8)
    ax_net.legend(loc='upper right', fontsize=8)
    ax_coupling.grid(True, alpha=0.3)
    
    print('[plots] Coupling and net torque plot created')
    row_offset += 1  # Add one more row offset for remaining plots
else:
    print('[plots] Warning: Coupling columns not found, skipping coupling plot')

# 11. Torque commands - Stacked visualization of components with blend ratio weighting (shifted)
ax14 = fig.add_subplot(gs[6 + row_offset, 0]) if row_offset > 0 else fig.add_subplot(gs[8, 0])
ax14.plot(t, df['torque_x'], 'b-', label='X Torque Cmd', linewidth=2, alpha=0.7)
ax14.plot(t, df['torque_y'], 'r-', label='Y Torque Cmd', linewidth=2, alpha=0.7)
ax14.plot(t, df['torque_z'], 'g-', label='Z Torque Cmd', linewidth=2, alpha=0.7)

# Add net torque (commanded - coupling) if coupling data exists
if all(col in df.columns for col in ['coupling_x', 'coupling_y', 'coupling_z']):
    net_torque_x = df['torque_x'] - df['coupling_x']  # What motors produce
    net_torque_y = df['torque_y'] - df['coupling_y']
    net_torque_z = df['torque_z'] - df['coupling_z']
    ax14.plot(t, net_torque_x, 'b--', label='X Net Torque', linewidth=1.5, alpha=0.5)
    ax14.plot(t, net_torque_y, 'r--', label='Y Net Torque', linewidth=1.5, alpha=0.5)
    ax14.plot(t, net_torque_z, 'g--', label='Z Net Torque', linewidth=1.5, alpha=0.5)
    print('[plots] Net torque (cmd - coupling) plotted')

ax14.set_title('Torque Commands (Solid) vs Net Torque (Dashed)')
ax14.set_xlabel('Time (s)')
ax14.set_ylabel('Torque (N·m)')
ax14.legend(loc='upper left', fontsize=7, ncol=2)
ax14.grid(True, alpha=0.3)

# 12. Control gains (shifted)
ax15 = fig.add_subplot(gs[6 + row_offset, 1]) if row_offset > 0 else fig.add_subplot(gs[8, 1])
ax15.plot(t, df['kp_att'], 'b-', label = 'Kp Attitude', linewidth=2)
ax15.plot(t, df['kd_att'], 'r-', label = 'Kd Attitude', linewidth=2)
ax15.plot(t, df['kp_rate'], 'g-', label= 'Kp Rate', linewidth=2)
ax15.plot(t, df['kd_rate'], 'm-', label= 'Kd Rate', linewidth=2)
ax15.set_xlabel('Time (s)')
ax15.set_ylabel('Gain Value')
ax15.set_title('Control Gains')
ax15.legend()
ax15.grid(True)

ax16a = fig.add_subplot(gs[6 + row_offset, 2]) if row_offset > 0 else fig.add_subplot(gs[8, 2])
pos_error_mag = np.sqrt(df['pos_err_x']**2 + df['pos_err_y']**2 + df['pos_err_z']**2)
ax16a.plot(t, pos_error_mag, 'k-', linewidth=2)
ax16a.set_xlabel('Time (s)', fontsize=10)
ax16a.set_ylabel('Position Error Magnitude (m)', fontsize=10)
ax16a.set_title('Position Error Magnitude', fontsize=11)
ax16a.grid(True)

# 13. XY Trajectory (shifted)
ax14a = fig.add_subplot(gs[7 + row_offset, 0]) if row_offset > 0 else fig.add_subplot(gs[9, 0])
ax14a.plot(df['pos_x'], df['pos_y'], 'b-', label='Actual XY', linewidth=2)
ax14a.plot(df['pos_des_x'], df['pos_des_y'], 'r--', label='Desired XY', linewidth=2)
ax14a.scatter(df['pos_x'].iloc[0], df['pos_y'].iloc[0], c='g', s=100, label='Start')
ax14a.scatter(df['pos_x'].iloc[-1], df['pos_y'].iloc[-1], c='r', s=100, label='End')
ax14a.set_xlabel('X (m)')
ax14a.set_ylabel('Y (m)')
ax14a.set_title('XY Trajectory')
ax14a.legend()
ax14a.grid(True)
ax14b = fig.add_subplot(gs[7 + row_offset, 1:]) if row_offset > 0 else fig.add_subplot(gs[9, 1:])
ax14b.plot(t, df['pos_z'], 'b-', label='Actual Z', linewidth=2)
ax14b.plot(t, df['pos_des_z'], 'r--', label='Desired Z', linewidth=2)
ax14b.set_xlabel('Time (s)')
ax14b.set_ylabel('Z Position (m)')
ax14b.set_title('Altitude vs Time')
ax14b.legend()
ax14b.grid(True)

# 14. Performance metrics (shifted)
ax15 = fig.add_subplot(gs[8 + row_offset, 0]) if row_offset > 0 else fig.add_subplot(gs[10, 0])
att_error_mag = np.sqrt(df['att_err_roll']**2 + df['att_err_pitch']**2)
ax15.plot(t, np.degrees(att_error_mag), 'k-', linewidth=2)
ax15.set_xlabel('Time (s)', fontsize=10)
ax15.set_ylabel('Attitude Error Magnitude (deg)', fontsize=10)
ax15.set_title('Attitude Error Magnitude', fontsize=11)
ax15.grid(True)

ax16 = fig.add_subplot(gs[8 + row_offset, 1]) if row_offset > 0 else fig.add_subplot(gs[10, 1])
rate_error_mag = np.sqrt(df['rate_err_p']**2 + df['rate_err_q']**2 + df['rate_err_r']**2)
ax16.plot(t, np.degrees(rate_error_mag), 'k-', linewidth=2)
ax16.set_xlabel('Time (s)', fontsize=10)
ax16.set_ylabel('Rate Error Magnitude (deg/s)', fontsize=10)
ax16.set_title('Rate Error Magnitude', fontsize=11)
ax16.grid(True)

# 15. Saturation indicators (shifted)
ax18 = fig.add_subplot(gs[8 + row_offset, 2]) if row_offset > 0 else fig.add_subplot(gs[10, 2])
sat_data = df[['rate_sat_p', 'rate_sat_q', 'rate_sat_r', 'acc_sat_x', 'acc_sat_y']].astype(int)
for i, col in enumerate(sat_data.columns):
    ax18.fill_between(t, i, i+sat_data[col], alpha=0.7, label=col)
ax18.set_xlabel('Time (s)', fontsize=10)
ax18.set_ylabel('Saturation Flags', fontsize=10)
ax18.set_title('Control Saturation Indicators', fontsize=11)
ax18.legend(fontsize=8)
ax18.grid(True)

# Place thrust subplot at the last grid row to avoid overlap
_nrows = gs.get_geometry()[0] 
ax13 = fig.add_subplot(gs[_nrows - 1, :])
ax13.plot(t, df['thrust_x'], 'b-', label='X Thrust', linewidth=2)
ax13.plot(t, df['thrust_y'], 'r-', label='Y Thrust', linewidth=2)
ax13.plot(t, df['thrust_z'], 'g-', label='Z Thrust', linewidth=2)
ax13.set_xlabel('Time (s)')
ax13.set_ylabel('Thrust Command')

ax13.legend()
ax13.grid(True)

# --- Improved Torque Ratio Plot: Using experiment torque and finite-difference angular acceleration ---
required_cols_ratio = ['p', 'q', 'r', 'exp_torque_x', 'exp_torque_y', 'torque_x', 'torque_y', 'phase_type']
if all(col in df.columns for col in required_cols_ratio):
    # Config inertia values (default fallback if missing)
    Ix = _get(cfg, ['inertia', 'Ix'], 0.02166666666666667)
    Iy = _get(cfg, ['inertia', 'Iy'], 0.02166666666666667)
    # We skip yaw ratio for clarity in this diagnostic

    # Body rates (logged in rad/s) -> ensure numeric
    p_rate = df['p'].to_numpy()
    q_rate = df['q'].to_numpy()
    time_arr = t.to_numpy()

    # Finite-difference angular accelerations (central difference for interior points)
    def central_diff(y, x):
        dydt = np.zeros_like(y)
        # Interior
        dydt[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
        # Ends (forward/backward difference)
        if len(y) > 1:
            dydt[0] = (y[1] - y[0]) / (x[1] - x[0])
            dydt[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
        return dydt

    p_dot_est = central_diff(p_rate, time_arr)
    q_dot_est = central_diff(q_rate, time_arr)

    # Low-pass smoothing (simple moving average) to reduce noise
    sample_rate = len(df) / time_arr[-1] if time_arr[-1] > 0 else 1000.0
    window_sec = 0.04  # 40 ms window
    Nw = max(5, int(sample_rate * window_sec))
    if Nw % 2 == 0:
        Nw += 1
    kernel = np.ones(Nw) / Nw
    def smooth(a):
        if len(a) < Nw:
            return a
        return np.convolve(a, kernel, mode='same')

    p_dot_f = smooth(p_dot_est)
    q_dot_f = smooth(q_dot_est)

    # Estimated torque from dynamics: tau = I * omega_dot
    tau_est_x = Ix * p_dot_f
    tau_est_y = Iy * q_dot_f

    # Use experiment (injected) torque ONLY for ratio baseline (exclude PD counter-torque)
    tau_cmd_x_exp = df['exp_torque_x'].to_numpy()
    tau_cmd_y_exp = df['exp_torque_y'].to_numpy()

    # Mask: only during active experiment phases (phase_type >= 0) and when |exp torque| above threshold
    phase_active_mask = df['phase_type'].to_numpy() >= 0
    torque_thresh = 0.01  # Nm threshold to avoid division near zero
    valid_x_mask = phase_active_mask & (np.abs(tau_cmd_x_exp) > torque_thresh)
    valid_y_mask = phase_active_mask & (np.abs(tau_cmd_y_exp) > torque_thresh)

    # Ratio: delivered / commanded (experiment component)
    ratio_x = np.full_like(tau_est_x, np.nan, dtype=float)
    ratio_y = np.full_like(tau_est_y, np.nan, dtype=float)
    ratio_x[valid_x_mask] = tau_est_x[valid_x_mask] / tau_cmd_x_exp[valid_x_mask]
    ratio_y[valid_y_mask] = tau_est_y[valid_y_mask] / tau_cmd_y_exp[valid_y_mask]

    # Optional robust scaling via local window least squares (for continuity)
    def window_scale(tau_est, tau_cmd, mask, window_samples= int(sample_rate*0.25)):
        scale = np.full_like(tau_est, np.nan, dtype=float)
        n = len(tau_est)
        w = max(10, window_samples)
        for i in range(n):
            if not mask[i]:
                continue
            a = max(0, i - w//2)
            b = min(n, i + w//2)
            sel = mask[a:b]
            if sel.sum() < 5:
                continue
            y = tau_est[a:b][sel]
            u = tau_cmd[a:b][sel]
            denom = np.dot(u, u)
            if denom < 1e-6:
                continue
            scale[i] = np.dot(u, y) / denom
        return smooth(scale)

    scale_x = window_scale(tau_est_x, tau_cmd_x_exp, valid_x_mask)
    scale_y = window_scale(tau_est_y, tau_cmd_y_exp, valid_y_mask)

    # Plot
    ax_ratio = fig.add_subplot(gs[9 + row_offset, :]) if row_offset > 0 else fig.add_subplot(gs[9, :])
    ax_ratio_r = ax_ratio.twinx()

    ln_rx, = ax_ratio.plot(t, ratio_x, 'b.', markersize=2, alpha=0.6, label='Instant Ratio X')
    ln_ry, = ax_ratio_r.plot(t, ratio_y, 'r.', markersize=2, alpha=0.6, label='Instant Ratio Y')
    ln_sx, = ax_ratio.plot(t, scale_x, 'b-', linewidth=1.8, alpha=0.9, label='Window Scale X')
    ln_sy, = ax_ratio_r.plot(t, scale_y, 'r-', linewidth=1.8, alpha=0.9, label='Window Scale Y')
    ax_ratio.axhline(1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.8, label='Ideal 1.0')

    # Global least squares scaling (overall effective gain)
    def global_scale(tau_est, tau_cmd, mask):
        sel_est = tau_est[mask]
        sel_cmd = tau_cmd[mask]
        denom = np.dot(sel_cmd, sel_cmd)
        return np.dot(sel_cmd, sel_est) / denom if denom > 1e-6 else np.nan

    g_scale_x = global_scale(tau_est_x, tau_cmd_x_exp, valid_x_mask)
    g_scale_y = global_scale(tau_est_y, tau_cmd_y_exp, valid_y_mask)
    print(f"\n[torque_ratio] Global scale factors (delivered/commanded experiment torque):")
    print(f"  Roll (X): {g_scale_x:.4f}")
    print(f"  Pitch (Y): {g_scale_y:.4f}")

    ax_ratio.set_xlabel('Time (s)')
    ax_ratio.set_ylabel('X Axis Ratio')
    ax_ratio_r.set_ylabel('Y Axis Ratio')
    ax_ratio.set_title('Experiment Torque Delivery Ratio (I·ω̇ / τ_exp)')

    # Symmetric axis limits based on percentiles
    def sym_lim(arr):
        a = arr[~np.isnan(arr)]
        if a.size < 10:
            return (-2, 2)
        q1, q99 = np.percentile(a, [1, 99])
        m = max(abs(q1), abs(q99))
        return (-m*1.1, m*1.1)
    ax_ratio.set_ylim(*sym_lim(ratio_x))
    ax_ratio_r.set_ylim(*sym_lim(ratio_y))

    # Legends
    leg_left = ax_ratio.legend(handles=[ln_rx, ln_sx], loc='upper left', fontsize=8)
    leg_right = ax_ratio_r.legend(handles=[ln_ry, ln_sy], loc='upper right', fontsize=8)
    ax_ratio.add_artist(leg_left)
    ax_ratio.grid(True, alpha=0.3)
    row_offset += 1;
    # --- New scatter plot: commanded experiment torque vs delivered (estimated) torque ---
    # Use same valid masks; downsample for clarity if dense
    scatter_ds = 5  # take every 5th valid sample
    cmd_x_valid = tau_cmd_x_exp[valid_x_mask]
    est_x_valid = tau_est_x[valid_x_mask]
    cmd_y_valid = tau_cmd_y_exp[valid_y_mask]
    est_y_valid = tau_est_y[valid_y_mask]

    if cmd_x_valid.size > 0 or cmd_y_valid.size > 0:
        ax_scatter = fig.add_subplot(gs[9 + row_offset, :]) if row_offset > 0 else fig.add_subplot(gs[9, :])
        # Roll axis scatter
        if cmd_x_valid.size > 0:
            # idx_x = np.arange(cmd_x_valid.size)[::scatter_ds]
            # ax_scatter.scatter(cmd_x_valid[idx_x], est_x_valid[idx_x], c='b', s=12, alpha=0.6, label='Roll Samples')
            # Regression through origin (global scale already computed)
            min_x, max_x = np.min(cmd_x_valid), np.max(cmd_x_valid)
            x_line = np.linspace(min_x, max_x, 100)
            ax_scatter.plot(x_line, x_line, 'k--', linewidth=1.0, alpha=0.7, label='Identity (y=x)')
            ax_scatter.plot(x_line, g_scale_x * x_line, 'b-', linewidth=1.5, alpha=0.9, label=f'Roll Fit (slope={g_scale_x:.3f})')
        # Pitch axis scatter
        if cmd_y_valid.size > 0:
            # idx_y = np.arange(cmd_y_valid.size)[::scatter_ds]
            # ax_scatter.scatter(cmd_y_valid[idx_y], est_y_valid[idx_y], c='r', s=12, alpha=0.6, label='Pitch Samples')
            min_y, max_y = np.min(cmd_y_valid), np.max(cmd_y_valid)
            y_line = np.linspace(min_y, max_y, 100)
            ax_scatter.plot(y_line, g_scale_y * y_line, 'r-', linewidth=1.5, alpha=0.9, label=f'Pitch Fit (slope={g_scale_y:.3f})')

        ax_scatter.set_xlabel('Commanded Experiment Torque (Nm)')
        ax_scatter.set_ylabel('Estimated Delivered Torque (Nm)')
        ax_scatter.set_title('Torque Scaling: Delivered vs Commanded (Roll & Pitch)')
        ax_scatter.grid(True, alpha=0.3)
        ax_scatter.legend(loc='best', fontsize=8, ncol=2)
        print('[plots] Torque scaling scatter plot created')
    else:
        print('[plots] Skipping torque scaling scatter (no valid samples)')

    row_offset += 1
    print('[plots] Improved torque ratio plot created')
else:
    print('[plots] Warning: Skipping improved torque ratio plot (missing columns)')

# --- Add new diagnostic plots: Estimated vs Motor Torque, and Coupling comparison ---
nrows = gs.get_geometry()[0]

# NEW: Estimated torque (from angular acceleration) vs motor torque command
# tau_est = I * p_dot_meas (what we measure from rate changes)
# tau_motor = torque_x/y/z (what we command)
# Comparison shows how well motors deliver what we ask

if all(col in df.columns for col in ['p_dot_meas', 'q_dot_meas', 'r_dot_meas']):
    # Use inertia values from config if available, else default
    try:
        Ix_est = _get(cfg, ['inertia', 'Ix'], 0.02166666666666667)
        Iy_est = _get(cfg, ['inertia', 'Iy'], 0.02166666666666667)
        Iz_est = _get(cfg, ['inertia', 'Iz'], 0.04)
    except:
        Ix_est, Iy_est, Iz_est = 0.02166666666666667, 0.02166666666666667, 0.04
    
    # Estimate torque from measured angular acceleration: tau = I * omega_dot
    tau_est_x = Ix_est * np.radians(df['p_dot_meas'])
    tau_est_y = Iy_est * np.radians(df['q_dot_meas'])
    tau_est_z = Iz_est * np.radians(df['r_dot_meas'])
    
    # X axis: Estimated vs Motor Torque
    ax_est_x = fig.add_subplot(gs[nrows-3, 0])
    ax_est_x.plot(t, tau_est_x, 'b--', label='Estimated (I·p_dot)', linewidth=2, alpha=0.7)
    ax_est_x.plot(t, df['torque_x'], 'c-', label='Motor Command', linewidth=1.5, alpha=0.8)
    if all(col in df.columns for col in ['coupling_x']):
        ax_est_x.plot(t, df['coupling_x'], 'k:', label='Coupling', linewidth=1, alpha=0.6)
    ax_est_x.set_ylabel('Torque X (N·m)')
    ax_est_x.set_title('Roll Axis: Estimated vs Motor Torque')
    ax_est_x.legend(fontsize=8)
    ax_est_x.grid(True, alpha=0.3)
    
    # Y axis: Estimated vs Motor Torque
    ax_est_y = fig.add_subplot(gs[nrows-3, 1])
    ax_est_y.plot(t, tau_est_y, 'r--', label='Estimated (I·q_dot)', linewidth=2, alpha=0.7)
    ax_est_y.plot(t, df['torque_y'], 'm-', label='Motor Command', linewidth=1.5, alpha=0.8)
    if all(col in df.columns for col in ['coupling_y']):
        ax_est_y.plot(t, df['coupling_y'], 'k:', label='Coupling', linewidth=1, alpha=0.6)
    ax_est_y.set_ylabel('Torque Y (N·m)')
    ax_est_y.set_title('Pitch Axis: Estimated vs Motor Torque')
    ax_est_y.legend(fontsize=8)
    ax_est_y.grid(True, alpha=0.3)
    
    # Z axis: Estimated vs Motor Torque
    ax_est_z = fig.add_subplot(gs[nrows-3, 2])
    ax_est_z.plot(t, tau_est_z, 'g--', label='Estimated (I·r_dot)', linewidth=2, alpha=0.7)
    ax_est_z.plot(t, df['torque_z'], 'darkgreen', label='Motor Command', linewidth=1.5, alpha=0.8)
    if all(col in df.columns for col in ['coupling_z']):
        ax_est_z.plot(t, df['coupling_z'], 'k:', label='Coupling', linewidth=1, alpha=0.6)
    ax_est_z.set_ylabel('Torque Z (N·m)')
    ax_est_z.set_title('Yaw Axis: Estimated vs Motor Torque')
    ax_est_z.legend(fontsize=8)
    ax_est_z.grid(True, alpha=0.3)
    
    print('[plots] Estimated vs Motor torque plots created')
else:
    print('[plots] Warning: Cannot create estimated torque plots (missing angular acceleration or inertia data)')

# NEW: Coupling torque analysis - Expected (omega x Iw) vs Actual coupling effect
if all(col in df.columns for col in ['coupling_x', 'coupling_y', 'coupling_z', 'p', 'q', 'r']):
    # Calculate expected coupling torque from body rates
    # Expected: tau_coupling = omega × (I × omega)
    # where omega = [p, q, r] in body frame
    
    p_rad = np.radians(df['p'])
    q_rad = np.radians(df['q'])
    r_rad = np.radians(df['r'])
    
    # Inertia matrix diagonal elements
    try:
        Ix = _get(cfg, ['inertia', 'Ix'], 0.02166666666666667)
        Iy = _get(cfg, ['inertia', 'Iy'], 0.02166666666666667)
        Iz = _get(cfg, ['inertia', 'Iz'], 0.04)
    except:
        Ix, Iy, Iz = 0.02166666666666667, 0.02166666666666667, 0.04
    
    # Expected coupling (theoretical: omega × (I * omega))
    # tau_c = [q*r*(Iz-Iy), p*r*(Ix-Iz), p*q*(Iy-Ix)]
    coupling_theoretical_x = q_rad * r_rad * (Iz - Iy)/Ix
    coupling_theoretical_y = p_rad * r_rad * (Ix - Iz)/Iy
    coupling_theoretical_z = p_rad * q_rad * (Iy - Ix)/Iz
    
    # Observed coupling: The actual torque applied is: tau_net = tau_commanded + tau_coupling
    # Therefore: tau_coupling_observed = tau_estimated - tau_commanded
    # Where tau_estimated = I * omega_dot_meas (from measured acceleration)
    # This captures all unmodeled effects, delays, saturation, etc.
    tau_estimated_x = Ix * np.radians(df['p_dot_meas'])
    tau_estimated_y = Iy * np.radians(df['q_dot_meas'])
    tau_estimated_z = Iz * np.radians(df['r_dot_meas'])
    
    # Residual: what torque actually happened minus what we commanded
    # If positive: system produced more torque than commanded (coupling helped)
    # If negative: system produced less torque than commanded (coupling opposed us)
    coupling_observed_x = tau_estimated_x - df['torque_x']
    coupling_observed_y = tau_estimated_y - df['torque_y']
    coupling_observed_z = tau_estimated_z - df['torque_z']
    
    # Plot comparison: Theoretical vs Observed coupling
    ax_coup = fig.add_subplot(gs[nrows-2, :])
    
    # Theoretical coupling (solid lines)
    ax_coup.plot(t, -50*coupling_theoretical_x, 'b-', label='Theoretical Coupling X', linewidth=2, alpha=0.7)
    ax_coup.plot(t, -50*coupling_theoretical_y, 'r-', label='Theoretical Coupling Y', linewidth=2, alpha=0.7)
    ax_coup.plot(t, -50*coupling_theoretical_z, 'g-', label='Theoretical Coupling Z', linewidth=2, alpha=0.7)
    
    # Observed coupling (dashed lines)
    ax_coup.plot(t, coupling_observed_x, 'b--', label='Observed Coupling X', linewidth=1.5, alpha=0.8)
    ax_coup.plot(t, coupling_observed_y, 'r--', label='Observed Coupling Y', linewidth=1.5, alpha=0.8)
    ax_coup.plot(t, coupling_observed_z, 'g--', label='Observed Coupling Z', linewidth=1.5, alpha=0.8)
    
    ax_coup.set_ylabel('Coupling Torque ω×(I·ω) (N·m)')
    ax_coup.set_xlabel('Time (s)')
    ax_coup.set_title('Coupling Torque: Theoretical (ω×Iω, solid) vs Observed (τ_est - τ_cmd, dashed)')
    ax_coup.legend(fontsize=8, loc='upper left', ncol=3)
    ax_coup.grid(True, alpha=0.3)
    
    # Calculate and print coupling estimation error
    coupling_error_x = np.sqrt(np.mean((coupling_observed_x - coupling_theoretical_x)**2))
    coupling_error_y = np.sqrt(np.mean((coupling_observed_y - coupling_theoretical_y)**2))
    coupling_error_z = np.sqrt(np.mean((coupling_observed_z - coupling_theoretical_z)**2))
    print(f'[plots] Coupling RMSE (Observed - Theoretical):')
    print(f'  X: {coupling_error_x:.6f} N·m')
    print(f'  Y: {coupling_error_y:.6f} N·m')
    print(f'  Z: {coupling_error_z:.6f} N·m')
    
    # Additional diagnostics: check correlation and timing offsets
    print(f'\n[plots] Coupling Correlation Analysis:')
    # Calculate steady-state stats (after 5 seconds)
    ss_mask = t >= 5.0
    if ss_mask.any():
        corr_x = np.corrcoef(coupling_theoretical_x[ss_mask], coupling_observed_x[ss_mask])[0, 1]
        corr_y = np.corrcoef(coupling_theoretical_y[ss_mask], coupling_observed_y[ss_mask])[0, 1]
        corr_z = np.corrcoef(coupling_theoretical_z[ss_mask], coupling_observed_z[ss_mask])[0, 1]
        print(f'  Correlation (t>=5s): X={corr_x:.3f}, Y={corr_y:.3f}, Z={corr_z:.3f}')
        
        # Mean values
        print(f'  Theoretical coupling mean (N·m):')
        print(f'    X: {coupling_theoretical_x[ss_mask].mean():.6f}')
        print(f'    Y: {coupling_theoretical_y[ss_mask].mean():.6f}')
        print(f'    Z: {coupling_theoretical_z[ss_mask].mean():.6f}')
        print(f'  Observed coupling mean (N·m):')
        print(f'    X: {coupling_observed_x[ss_mask].mean():.6f}')
        print(f'    Y: {coupling_observed_y[ss_mask].mean():.6f}')
        print(f'    Z: {coupling_observed_z[ss_mask].mean():.6f}')
    
else:
    print('[plots] Warning: Cannot create coupling comparison plot (missing coupling or rate data)')

# NEW: Commanded vs Allocated Torque Plots (from ControlAllocatorStatus)
# Shows what we asked for vs what PX4 could actually deliver
has_alloc_data = all(col in df.columns for col in ['unallocated_torque_x', 'unallocated_torque_y', 'unallocated_torque_z',
                                                     'allocated_torque_x', 'allocated_torque_y', 'allocated_torque_z',
                                                     'torque_setpoint_achieved'])

if has_alloc_data:
    # Row 1: Commanded vs Allocated for all axes
    ax_alloc = fig.add_subplot(gs[nrows-4, :])
    
    # X axis (Roll) - Commanded vs Allocated
    ax_alloc.plot(t, df['torque_x'], 'b-', label='Commanded X', linewidth=2, alpha=0.8)
    ax_alloc.plot(t, df['allocated_torque_x'], 'b--', label='Allocated X', linewidth=1.5, alpha=0.7)
    
    # Y axis (Pitch) - Commanded vs Allocated  
    ax_alloc.plot(t, df['torque_y'], 'r-', label='Commanded Y', linewidth=2, alpha=0.8)
    ax_alloc.plot(t, df['allocated_torque_y'], 'r--', label='Allocated Y', linewidth=1.5, alpha=0.7)
    
    # Z axis (Yaw) - Commanded vs Allocated
    ax_alloc.plot(t, df['torque_z'], 'g-', label='Commanded Z', linewidth=2, alpha=0.8)
    ax_alloc.plot(t, df['allocated_torque_z'], 'g--', label='Allocated Z', linewidth=1.5, alpha=0.7)
    
    ax_alloc.set_ylabel('Torque (N·m)')
    ax_alloc.set_xlabel('Time (s)')
    ax_alloc.set_title('Commanded (solid) vs Allocated (dashed) Torques - PX4 Control Allocation')
    ax_alloc.legend(fontsize=9, loc='upper right', ncol=3)
    ax_alloc.grid(True, alpha=0.3)
    
    # Row 2: Unallocated torques (what couldn't be achieved) and setpoint achieved flag
    ax_unalloc = fig.add_subplot(gs[nrows-5, :])
    ax_unalloc_flag = ax_unalloc.twinx()
    
    # Unallocated torques
    ax_unalloc.plot(t, df['unallocated_torque_x'], 'b-', label='Unallocated X', linewidth=2, alpha=0.8)
    ax_unalloc.plot(t, df['unallocated_torque_y'], 'r-', label='Unallocated Y', linewidth=2, alpha=0.8)
    ax_unalloc.plot(t, df['unallocated_torque_z'], 'g-', label='Unallocated Z', linewidth=2, alpha=0.8)
    ax_unalloc.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Torque setpoint achieved flag (binary)
    ax_unalloc_flag.fill_between(t, 0, df['torque_setpoint_achieved'].astype(int), 
                                  alpha=0.2, color='purple', label='Setpoint Achieved')
    ax_unalloc_flag.set_ylim(-0.1, 1.1)
    ax_unalloc_flag.set_ylabel('Setpoint Achieved', color='purple')
    ax_unalloc_flag.tick_params(axis='y', labelcolor='purple')
    
    ax_unalloc.set_ylabel('Unallocated Torque (N·m)')
    ax_unalloc.set_xlabel('Time (s)')
    ax_unalloc.set_title('Unallocated Torque (Commanded - Allocated) & Setpoint Achievement Status')
    ax_unalloc.legend(fontsize=9, loc='upper left')
    ax_unalloc_flag.legend(fontsize=9, loc='upper right')
    ax_unalloc.grid(True, alpha=0.3)
    
    # Calculate and print allocation statistics
    unalloc_mag = np.sqrt(df['unallocated_torque_x']**2 + df['unallocated_torque_y']**2 + df['unallocated_torque_z']**2)
    pct_achieved = df['torque_setpoint_achieved'].mean() * 100
    
    print(f'\n[plots] Torque Allocation Statistics:')
    print(f'  Setpoint achieved: {pct_achieved:.1f}% of the time')
    print(f'  Unallocated torque RMS: X={df["unallocated_torque_x"].abs().mean():.4f}, Y={df["unallocated_torque_y"].abs().mean():.4f}, Z={df["unallocated_torque_z"].abs().mean():.4f} N·m')
    print(f'  Max unallocated: X={df["unallocated_torque_x"].abs().max():.4f}, Y={df["unallocated_torque_y"].abs().max():.4f}, Z={df["unallocated_torque_z"].abs().max():.4f} N·m')
    print(f'  Unallocated magnitude mean: {unalloc_mag.mean():.4f} N·m')
    print('[plots] Commanded vs Allocated torque plots created')
else:
    print('[plots] Warning: Torque allocation columns not found (unallocated_torque_*, allocated_torque_*, torque_setpoint_achieved)')
    print('[plots] These are available from ControlAllocatorStatus in newer logs')

# Save plots with unique names per log file
base_name = os.path.splitext(os.path.basename(latest_log))[0]

# Determine output directory - use source-specific subdirectory if specified
base_outputs_dir = '/home/pyro/ws_offboard_control/flight_data/outputs'
if args.output_dir:
    outputs_dir = args.output_dir
elif args.source:
    outputs_dir = os.path.join(base_outputs_dir, args.source)
else:
    outputs_dir = base_outputs_dir
os.makedirs(outputs_dir, exist_ok=True)
pdf_out = os.path.join(outputs_dir, f'{base_name}_analysis.pdf')

# Add source info to title if available
title_suffix = f' ({args.source})' if args.source else ''
plt.suptitle(f'Comprehensive Flight Data Analysis{title_suffix}', fontsize=16, y=0.98)

# Add a compact config summary at the bottom of the page
if cfg is not None:
    footer = (
        f"Att[kp={_get(cfg,['attitude','kp'], _get(cfg,['attitude','kp_base'],4.0))},kd={_get(cfg,['attitude','kd'], _get(cfg,['attitude','kd_base'],0.0))}] "
        f"Sched={_get(cfg,['gain_scheduling','enabled'],False)}@{_get(cfg,['gain_scheduling','rate_threshold'],5.0)} "
        f"NDI[kp={_get(cfg,['ndi_rate','kp'],0.1)},kd={_get(cfg,['ndi_rate','kd'],0.0)}] "
        f"INDI[en={_get(cfg,['indi','enabled'],False)},scale={_get(cfg,['indi','scale'],0.75)},blend={_get(cfg,['indi','blend_ratio'],0.3)}] "
        f"YawFF[dt={_get(cfg,['yaw_ff','dt_base'],0.035)}+{_get(cfg,['yaw_ff','dt_gain'],0.008)}|max={_get(cfg,['yaw_ff','dt_max'],0.1)}]"
    )
else:
    footer = "No configuration found"
fig.text(0.01, 0.005, "\n".join(textwrap.wrap(footer, width=180)), fontsize=15, family='monospace', transform=fig.transFigure)
plt.savefig(pdf_out, bbox_inches='tight')
print(f'Plots saved to {pdf_out}')

# Save config summary text alongside plots for traceability
try:
    summary_out = os.path.join(outputs_dir, f'{base_name}_config_summary.txt')
    with open(summary_out, 'w') as f:
        if cfg_source:
            f.write(f'CONFIG SOURCE: {cfg_source}\n\n')
        f.write(config_summary + '\n')
    print(f'Config summary saved to {summary_out}')
except Exception as e:
    print(f'Failed to save config summary: {e}')


# Statistics
print('\n=== FLIGHT STATISTICS ===')
print(f'Flight duration: {t.iloc[-1]:.2f} seconds')
print(f'Position error RMS: {np.sqrt(np.mean(pos_error_mag**2)):.3f} m')
print(f'Attitude error RMS: {np.sqrt(np.mean(att_error_mag**2))*180/np.pi:.3f} deg')
print(f'Rate error RMS: {np.sqrt(np.mean(rate_error_mag**2))*180/np.pi:.3f} deg/s')
print(f'Max position error: {pos_error_mag.max():.3f} m')
print(f'Max attitude error: {att_error_mag.max()*180/np.pi:.3f} deg')
print(f'Max rate error: {rate_error_mag.max()*180/np.pi:.3f} deg/s')
print(f'Data points: {len(df)}')
print(f'Sampling rate: {len(df)/t.iloc[-1]:.1f} Hz')
