#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import json
import textwrap

# Read data - find the most recent flight log
import glob
import os
log_files = glob.glob('/home/pyro/ws_offboard_control/flight_data/logs/flight_log_*.csv')
if not log_files:
    print('No flight log files found!')
    exit(1)
latest_log = max(log_files, key=os.path.getctime)
print(f'Reading log file: {latest_log}')
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
fig = plt.figure(figsize=(40, 40))  # Increased height for more rows
gs = GridSpec(14, 3, figure=fig, hspace=0.3, wspace=0.3)  # 14 rows: angular accel + coupling plots

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

# 3. Attitude tracking
ax4 = fig.add_subplot(gs[1, 2])
ax4.plot(t, np.degrees(df['roll']), 'b-', label='Roll', linewidth=2)
ax4.plot(t, np.degrees(df['pitch']), 'r-', label='Pitch', linewidth=2)
ax4.plot(t, np.degrees(df['yaw']), 'g-', label='Yaw', linewidth=2)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Attitude (deg)')
ax4.set_title('Attitude Angles')
ax4.legend()
ax4.grid(True)

# 4. Angular rates
ax5 = fig.add_subplot(gs[2, 0])
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

ax6 = fig.add_subplot(gs[2, 1])
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

ax7 = fig.add_subplot(gs[2, 2])
ax7.plot(t, np.degrees(df['r']), 'g-', label='Actual r', linewidth=2)
ax7.plot(t, np.degrees(df['r_des']), 'g--', label='Desired r', linewidth=2)
ax7.set_xlabel('Time (s)')
ax7.set_ylabel('Yaw Rate (deg/s)')
ax7.set_title('Yaw Rate Tracking')
ax7.legend()
ax7.grid(True)

# 5. p Error (full row)
ax10 = fig.add_subplot(gs[3, :])
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
ax11 = fig.add_subplot(gs[4, :])
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
ax9 = fig.add_subplot(gs[5, :])
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
    ax_pdot = fig.add_subplot(gs[6, :])
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
    ax_qdot = fig.add_subplot(gs[7, :])
    ax_qdot_torque = ax_qdot.twinx()
    
    ax_qdot.plot(t, np.degrees(df['q_dot_des']), 'b--', linewidth=0.5, label='Desired q_dot', alpha=0.8)
    ax_qdot.plot(t, np.degrees(df['q_dot_meas']), 'r-', linewidth=0.5, label='Measured q_dot')
    
    # # Stacked torque components for Y axis
    # if all(col in df.columns for col in ['tau_ndi_y', 'tau_indi_y', 'indi_blend_ratio']):
    #     tau_ndi_weighted_y = df['tau_ndi_y'] * (1.0 - df['indi_blend_ratio'])
    #     tau_indi_weighted_y = df['tau_indi_y'] * df['indi_blend_ratio']
    #     tau_fb_y = df['torque_y'] - tau_ndi_weighted_y - tau_indi_weighted_y
        
    #     ax_qdot_torque.fill_between(t, 0, tau_ndi_weighted_y, label='NDI', color='purple', alpha=0.4, linewidth=0)
    #     ax_qdot_torque.fill_between(t, tau_ndi_weighted_y, tau_ndi_weighted_y + tau_indi_weighted_y, 
    #                                  label='INDI', color='orange', alpha=0.4, linewidth=0)
    #     ax_qdot_torque.fill_between(t, tau_ndi_weighted_y + tau_indi_weighted_y, df['torque_y'], 
    #                                  label='Feedback', color='pink', alpha=0.4, linewidth=0)
    #     ax_qdot_torque.plot(t, df['torque_y'], 'm-', linewidth=0.5, alpha=0.8, label='Total Y Torque')
    # else:
    ax_qdot_torque.plot(t, df['torque_y'], 'm-', linewidth=0.1, alpha=0.7, label='Y Torque')
    
    # Add coupling and net torque overlays if available
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
    ax_rdot = fig.add_subplot(gs[8, :])
    ax_rdot_torque = ax_rdot.twinx()
    
    # ax_rdot.plot(t, np.degrees(df['r_dot_des']), 'r--', linewidth=0.5, label='Desired r_dot', alpha=0.8)
    # ax_rdot.plot(t, np.degrees(df['r_dot_meas']), 'g-', linewidth=0.5, label='Measured r_dot')
    
    # Stacked torque components for Z axis
    if all(col in df.columns for col in ['tau_ndi_z', 'tau_indi_z', 'indi_blend_ratio']):
        tau_ndi_weighted_z = df['tau_ndi_z'] * (1.0 - df['indi_blend_ratio'])
        tau_indi_weighted_z = df['tau_indi_z'] * df['indi_blend_ratio']
        tau_fb_z = df['torque_z'] - tau_ndi_weighted_z - tau_indi_weighted_z
        
        ax_rdot_torque.fill_between(t, 0, tau_ndi_weighted_z, label='NDI', color='purple', alpha=0.4, linewidth=0)
        ax_rdot_torque.fill_between(t, tau_ndi_weighted_z, tau_ndi_weighted_z + tau_indi_weighted_z, 
                                     label='INDI', color='yellow', alpha=0.4, linewidth=0)
        ax_rdot_torque.fill_between(t, tau_ndi_weighted_z + tau_indi_weighted_z, df['torque_z'], 
                                     label='Feedback', color='lightgreen', alpha=0.4, linewidth=0)
        # ax_rdot_torque.plot(t, df['torque_z'], 'purple', linewidth=0.01, alpha=0.8, label='Total Z Torque')
    else:
        ax_rdot_torque.plot(t, df['torque_z'], 'g-', linewidth=0.1, alpha=0.7, label='Z Torque')
    
    # Align axes
    if steady_state_mask.any():
        rdot_max = max(abs(np.degrees(df.loc[steady_state_mask, 'r_dot_des']).max()),
                       abs(np.degrees(df.loc[steady_state_mask, 'r_dot_des']).min()),
                       abs(np.degrees(df.loc[steady_state_mask, 'r_dot_meas']).max()),
                       abs(np.degrees(df.loc[steady_state_mask, 'r_dot_meas']).min()))
        torque_max = max(abs(df.loc[steady_state_mask, 'torque_z'].min()), 
                         abs(df.loc[steady_state_mask, 'torque_z'].max()))
    else:
        rdot_max = max(abs(np.degrees(df['r_dot_des']).max()), abs(np.degrees(df['r_dot_meas']).max()))
        torque_max = max(abs(df['torque_z'].min()), abs(df['torque_z'].max()))
    
    ax_rdot.set_ylim(-rdot_max * 1.1, rdot_max * 1.1)
    ax_rdot_torque.set_ylim(-torque_max * 1.1, torque_max * 1.1)
    ax_rdot.set_ylabel('Yaw Ang. Accel. (deg/s²)', color='g')
    ax_rdot_torque.set_ylabel('Z Torque (N·m)', color='darkgreen')
    ax_rdot.set_title('Yaw Angular Acceleration (r_dot) & Z Torque Command')
    ax_rdot.tick_params(axis='y', labelcolor='g')
    ax_rdot_torque.tick_params(axis='y', labelcolor='darkgreen')
    ax_rdot.legend(loc='upper left')
    ax_rdot_torque.legend(loc='upper right')
    ax_rdot.grid(True)
    
    print('[plots] Angular acceleration plots created successfully')
    
    # Shift remaining plots down by 3 rows
    row_offset = 3
else:
    print('[plots] Warning: Angular acceleration columns not found, skipping ang accel plots')
    row_offset = 0

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

# 8. Position errors
# ax8 = fig.add_subplot(gs[6, 0])
# ax8.plot(t, df['pos_err_x'], 'b-', label='X Error', linewidth=2)
# ax8.plot(t, df['pos_err_y'], 'r-', label='Y Error', linewidth=2)
# ax8.plot(t, df['pos_err_z'], 'g-', label='Z Error', linewidth=2)
# # Calculate y-limits from steady-state data
# if steady_state_mask.any():
#     pos_err_data_ss = pd.concat([df.loc[steady_state_mask, 'pos_err_x'], 
#                                   df.loc[steady_state_mask, 'pos_err_y'], 
#                                   df.loc[steady_state_mask, 'pos_err_z']])
#     pos_err_min, pos_err_max = pos_err_data_ss.min(), pos_err_data_ss.max()
#     pos_err_margin = (pos_err_max - pos_err_min) * 0.1
#     ax8.set_ylim(pos_err_min - pos_err_margin, pos_err_max + pos_err_margin)
# ax8.set_xlabel('Time (s)')
# ax8.set_ylabel('Position Error (m)')
# ax8.set_title('Position Errors')
# ax8.legend()
# ax8.grid(True)

# 9. r Rate error
# ax12 = fig.add_subplot(gs[6, 1])
# ax12.plot(t, np.degrees(df['rate_err_r']), 'g-', linewidth=2)
# # Calculate y-limits from steady-state data
# if steady_state_mask.any():
#     r_err_max = max(abs(np.degrees(df.loc[steady_state_mask, 'rate_err_r']).min()), 
#                     abs(np.degrees(df.loc[steady_state_mask, 'rate_err_r']).max()))
#     ax12.set_ylim(-r_err_max * 1.1, r_err_max * 1.1)
# ax12.set_xlabel('Time (s)')
# ax12.set_ylabel('Yaw Rate Error (deg/s)')
# ax12.set_title('Yaw Rate (r) Error')
# ax12.grid(True)

# 10. Thrust commands (shifted)
ax13 = fig.add_subplot(gs[6 + row_offset, :]) if row_offset > 0 else fig.add_subplot(gs[6, :])
ax13.plot(t, df['thrust_x'], 'b-', label='X Thrust', linewidth=2)
ax13.plot(t, df['thrust_y'], 'r-', label='Y Thrust', linewidth=2)
ax13.plot(t, df['thrust_z'], 'g-', label='Z Thrust', linewidth=2)
ax13.set_xlabel('Time (s)')
ax13.set_ylabel('Thrust Command')

# ax13.set_title('Thrust Commands')
ax13.legend()
ax13.grid(True)

# 11. Torque commands - Stacked visualization of components with blend ratio weighting (shifted)
ax14 = fig.add_subplot(gs[7 + row_offset, 0]) if row_offset > 0 else fig.add_subplot(gs[7, 0])
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
ax15 = fig.add_subplot(gs[7 + row_offset, 1]) if row_offset > 0 else fig.add_subplot(gs[7, 1])
ax15.plot(t, df['kp_att'], 'b-', label = 'Kp Attitude', linewidth=2)
ax15.plot(t, df['kd_att'], 'r-', label = 'Kd Attitude', linewidth=2)
ax15.plot(t, df['kp_rate'], 'g-', label= 'Kp Rate', linewidth=2)
ax15.plot(t, df['kd_rate'], 'm-', label= 'Kd Rate', linewidth=2)
ax15.set_xlabel('Time (s)')
ax15.set_ylabel('Gain Value')
ax15.set_title('Control Gains')
ax15.legend()
ax15.grid(True)

ax16a = fig.add_subplot(gs[7 + row_offset, 2]) if row_offset > 0 else fig.add_subplot(gs[7, 2])
pos_error_mag = np.sqrt(df['pos_err_x']**2 + df['pos_err_y']**2 + df['pos_err_z']**2)
ax16a.plot(t, pos_error_mag, 'k-', linewidth=2)
ax16a.set_xlabel('Time (s)', fontsize=10)
ax16a.set_ylabel('Position Error Magnitude (m)', fontsize=10)
ax16a.set_title('Position Error Magnitude', fontsize=11)
ax16a.grid(True)

# 13. XY Trajectory (shifted)
ax14a = fig.add_subplot(gs[8 + row_offset, 0]) if row_offset > 0 else fig.add_subplot(gs[8, 0])
ax14a.plot(df['pos_x'], df['pos_y'], 'b-', label='Actual XY', linewidth=2)
ax14a.plot(df['pos_des_x'], df['pos_des_y'], 'r--', label='Desired XY', linewidth=2)
ax14a.scatter(df['pos_x'].iloc[0], df['pos_y'].iloc[0], c='g', s=100, label='Start')
ax14a.scatter(df['pos_x'].iloc[-1], df['pos_y'].iloc[-1], c='r', s=100, label='End')
ax14a.set_xlabel('X (m)')
ax14a.set_ylabel('Y (m)')
ax14a.set_title('XY Trajectory')
ax14a.legend()
ax14a.grid(True)
ax14b = fig.add_subplot(gs[8 + row_offset, 1:]) if row_offset > 0 else fig.add_subplot(gs[8, 1:])
ax14b.plot(t, df['pos_z'], 'b-', label='Actual Z', linewidth=2)
ax14b.plot(t, df['pos_des_z'], 'r--', label='Desired Z', linewidth=2)
ax14b.set_xlabel('Time (s)')
ax14b.set_ylabel('Z Position (m)')
ax14b.set_title('Altitude vs Time')
ax14b.legend()
ax14b.grid(True)

# 14. Performance metrics (shifted)
ax15 = fig.add_subplot(gs[9 + row_offset, 0]) if row_offset > 0 else fig.add_subplot(gs[9, 0])
att_error_mag = np.sqrt(df['att_err_roll']**2 + df['att_err_pitch']**2)
ax15.plot(t, np.degrees(att_error_mag), 'k-', linewidth=2)
ax15.set_xlabel('Time (s)', fontsize=10)
ax15.set_ylabel('Attitude Error Magnitude (deg)', fontsize=10)
ax15.set_title('Attitude Error Magnitude', fontsize=11)
ax15.grid(True)

ax16 = fig.add_subplot(gs[9 + row_offset, 1]) if row_offset > 0 else fig.add_subplot(gs[9, 1])
rate_error_mag = np.sqrt(df['rate_err_p']**2 + df['rate_err_q']**2 + df['rate_err_r']**2)
ax16.plot(t, np.degrees(rate_error_mag), 'k-', linewidth=2)
ax16.set_xlabel('Time (s)', fontsize=10)
ax16.set_ylabel('Rate Error Magnitude (deg/s)', fontsize=10)
ax16.set_title('Rate Error Magnitude', fontsize=11)
ax16.grid(True)

# 15. Saturation indicators (shifted)
ax18 = fig.add_subplot(gs[9 + row_offset, 2]) if row_offset > 0 else fig.add_subplot(gs[9, 2])
sat_data = df[['rate_sat_p', 'rate_sat_q', 'rate_sat_r', 'acc_sat_x', 'acc_sat_y']].astype(int)
for i, col in enumerate(sat_data.columns):
    ax18.fill_between(t, i, i+sat_data[col], alpha=0.7, label=col)
ax18.set_xlabel('Time (s)', fontsize=10)
ax18.set_ylabel('Saturation Flags', fontsize=10)
ax18.set_title('Control Saturation Indicators', fontsize=11)
ax18.legend(fontsize=8)
ax18.grid(True)

# Save plots with unique names per log file
base_name = os.path.splitext(os.path.basename(latest_log))[0]
outputs_dir = '/home/pyro/ws_offboard_control/flight_data/outputs'
os.makedirs(outputs_dir, exist_ok=True)
pdf_out = os.path.join(outputs_dir, f'{base_name}_analysis.pdf')

plt.suptitle('Comprehensive Flight Data Analysis', fontsize=16, y=0.98)

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
