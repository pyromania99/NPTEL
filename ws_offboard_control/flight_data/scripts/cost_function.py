"""
Cost Function Components for Controller Tuning

1. TRACKING PERFORMANCE (40% weight)
   - Position RMS error
   - Attitude RMS error  
   - Rate tracking RMS error

2. CONTROL SMOOTHNESS (20% weight)
   - Derivative of thrust commands
   - Derivative of torque commands
   - Penalizes high-frequency oscillations

3. CONTROL EFFORT (10% weight)
   - Total control energy used
   - Prevents overly aggressive control

4. STABILITY (20% weight)
   - Variance in steady-state period
   - Convergence to setpoint

5. OSCILLATION PENALTY (10% weight)
   - High-frequency content in control signals
   - Second derivative of torque commands

TOTAL COST = w1*tracking + w2*smoothness + w3*effort + w4*stability + w5*oscillations

Lower cost = better performance
"""

import numpy as np

def compute_controller_cost(log_data):
    """
    Compute cost function for controller performance
    
    Args:
        log_data: numpy array with columns [time, states, errors, commands]
    
    Returns:
        float: Total cost (lower is better)
    """
    
    # Extract signals
    pos_errors = log_data[:, 7:10]     # [x_err, y_err, z_err]
    att_errors = log_data[:, 16:18]    # [roll_err, pitch_err]  
    rate_errors = log_data[:, 22:25]   # [p_err, q_err, r_err]
    thrust_cmds = log_data[:, 25:28]   # [thrust_x, thrust_y, thrust_z]
    torque_cmds = log_data[:, 28:31]   # [torque_x, torque_y, torque_z]
    
    # 1. Tracking Performance
    pos_rms = np.sqrt(np.mean(np.sum(pos_errors**2, axis=1)))
    att_rms = np.sqrt(np.mean(np.sum(att_errors**2, axis=1)))
    rate_rms = np.sqrt(np.mean(np.sum(rate_errors**2, axis=1)))
    
    tracking_cost = 1.0*pos_rms + 5.0*att_rms + 1.5*rate_rms
    
    # 2. Control Smoothness  
    thrust_smooth = np.mean(np.sum(np.diff(thrust_cmds, axis=0)**2, axis=1))
    torque_smooth = np.mean(np.sum(np.diff(torque_cmds, axis=0)**2, axis=1))
    
    smoothness_cost = 0.1*thrust_smooth + 0.2*torque_smooth
    
    # 3. Control Effort
    effort_cost = 0.05 * np.mean(np.sum(thrust_cmds**2 + torque_cmds**2, axis=1))
    
    # 4. Stability (steady-state variance)
    if len(log_data) > 100:
        steady_state = log_data[-50:, 7:10]  # Last 50 samples of position
        stability_cost = 0.5 * np.mean(np.var(steady_state, axis=0))
    else:
        stability_cost = 10.0
    
    # 5. Oscillation Penalty
    oscillation_cost = 0.1 * np.mean(np.sum(np.diff(torque_cmds, n=2, axis=0)**2, axis=1))
    
    total_cost = tracking_cost + smoothness_cost + effort_cost + stability_cost + oscillation_cost
    
    return total_cost, {
        'tracking': tracking_cost,
        'smoothness': smoothness_cost, 
        'effort': effort_cost,
        'stability': stability_cost,
        'oscillation': oscillation_cost
    }