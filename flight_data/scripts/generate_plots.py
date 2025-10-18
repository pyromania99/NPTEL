#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

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

# Create comprehensive figure with subplots
plt.style.use('seaborn-v0_8')
fig = plt.figure(figsize=(40, 28))
gs = GridSpec(10, 3, figure=fig, hspace=0.3, wspace=0.3)

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
ax10_torque.plot(t, df['torque_x'], 'c-', linewidth=1.5, alpha=0.7, label='X Torque')

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
ax11_torque.plot(t, df['torque_y'], 'm-', linewidth=1.5, alpha=0.7, label='Y Torque')

# Align zero lines - use steady-state data for limits
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

# 10. Thrust commands
ax13 = fig.add_subplot(gs[6, :])
ax13.plot(t, df['thrust_x'], 'b-', label='X Thrust', linewidth=2)
ax13.plot(t, df['thrust_y'], 'r-', label='Y Thrust', linewidth=2)
ax13.plot(t, df['thrust_z'], 'g-', label='Z Thrust', linewidth=2)
ax13.set_xlabel('Time (s)')
ax13.set_ylabel('Thrust Command')
# ax13.set_title('Thrust Commands')
ax13.legend()
ax13.grid(True)

# 11. Torque commands
ax14 = fig.add_subplot(gs[7, 0])
ax14.plot(t, df['torque_x'], 'b-', label='X Torque', linewidth=2)
ax14.plot(t, df['torque_y'], 'r-', label='Y Torque', linewidth=2)
ax14.plot(t, df['torque_z'], 'g-', label='Z Torque', linewidth=2)
ax14.set_xlabel('Time (s)')
ax14.set_ylabel('Torque Command')
# ax14.set_title('Torque Commands')
ax14.legend()
ax14.grid(True)

# 12. Control gains
ax15 = fig.add_subplot(gs[7, 1])
ax15.plot(t, df['kp_att'], 'b-', label='Kp Attitude', linewidth=2)
ax15.plot(t, df['kd_att'], 'r-', label='Kd Attitude', linewidth=2)
ax15.plot(t, df['kp_rate'], 'g-', label='Kp Rate', linewidth=2)
ax15.plot(t, df['kd_rate'], 'm-', label='Kd Rate', linewidth=2)
ax15.set_xlabel('Time (s)')
ax15.set_ylabel('Gain Value')
ax15.set_title('Control Gains')
ax15.legend()
ax15.grid(True)

ax16a = fig.add_subplot(gs[7, 2])
pos_error_mag = np.sqrt(df['pos_err_x']**2 + df['pos_err_y']**2 + df['pos_err_z']**2)
ax16a.plot(t, pos_error_mag, 'k-', linewidth=2)
ax16a.set_xlabel('Time (s)', fontsize=10)
ax16a.set_ylabel('Position Error Magnitude (m)', fontsize=10)
ax16a.set_title('Position Error Magnitude', fontsize=11)
ax16a.grid(True)

# 13. XY Trajectory
ax14a = fig.add_subplot(gs[8, 0])
ax14a.plot(df['pos_x'], df['pos_y'], 'b-', label='Actual XY', linewidth=2)
ax14a.plot(df['pos_des_x'], df['pos_des_y'], 'r--', label='Desired XY', linewidth=2)
ax14a.scatter(df['pos_x'].iloc[0], df['pos_y'].iloc[0], c='g', s=100, label='Start')
ax14a.scatter(df['pos_x'].iloc[-1], df['pos_y'].iloc[-1], c='r', s=100, label='End')
ax14a.set_xlabel('X (m)')
ax14a.set_ylabel('Y (m)')
ax14a.set_title('XY Trajectory')
ax14a.legend()
ax14a.grid(True)
ax14b = fig.add_subplot(gs[8, 1:])
ax14b.plot(t, df['pos_z'], 'b-', label='Actual Z', linewidth=2)
ax14b.plot(t, df['pos_des_z'], 'r--', label='Desired Z', linewidth=2)
ax14b.set_xlabel('Time (s)')
ax14b.set_ylabel('Z Position (m)')
ax14b.set_title('Altitude vs Time')
ax14b.legend()
ax14b.grid(True)

# 14. Performance metrics
ax15 = fig.add_subplot(gs[9, 0])
att_error_mag = np.sqrt(df['att_err_roll']**2 + df['att_err_pitch']**2)
ax15.plot(t, np.degrees(att_error_mag), 'k-', linewidth=2)
ax15.set_xlabel('Time (s)', fontsize=10)
ax15.set_ylabel('Attitude Error Magnitude (deg)', fontsize=10)
ax15.set_title('Attitude Error Magnitude', fontsize=11)
ax15.grid(True)

ax16 = fig.add_subplot(gs[9, 1])
rate_error_mag = np.sqrt(df['rate_err_p']**2 + df['rate_err_q']**2 + df['rate_err_r']**2)
ax16.plot(t, np.degrees(rate_error_mag), 'k-', linewidth=2)
ax16.set_xlabel('Time (s)', fontsize=10)
ax16.set_ylabel('Rate Error Magnitude (deg/s)', fontsize=10)
ax16.set_title('Rate Error Magnitude', fontsize=11)
ax16.grid(True)

# 15. Saturation indicators
ax18 = fig.add_subplot(gs[9, 2])
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
plt.savefig(pdf_out, bbox_inches='tight')
print(f'Plots saved to {pdf_out}')


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
