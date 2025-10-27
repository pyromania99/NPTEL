import numpy as np
import pandas as pd

# Path to your log file
log_path = '/home/pyro/ws_offboard_control/flight_data/logs/flight_log_latest.csv'

# Load log
log = pd.read_csv(log_path)

# Extract relevant columns (update these if your log uses different names)
# Euler rate errors (should be logged or computed as desired - measured)
p_dot_err = log['p_dot_des'].values if 'p_dot_des' in log else log['p_des'].values - log['p'].values
q_dot_err = log['q_dot_des'].values if 'q_dot_des' in log else log['q_des'].values - log['q'].values
r_dot_err = log['r_dot_des'].values if 'r_dot_des' in log else log['r_des'].values - log['r'].values

# Stack errors into (N, 3) array
err = np.vstack([p_dot_err, q_dot_err, r_dot_err]).T

# Compute error derivatives (finite difference)
err_dot = np.gradient(err, axis=0)

# mu3_des from log (should be the NDI/innerloop torque command)
mu3_x = log['tau_ndi_x'].values
mu3_y = log['tau_ndi_y'].values
mu3_z = log['tau_ndi_z'].values
mu3 = np.vstack([mu3_x, mu3_y, mu3_z]).T

# Fit each axis separately
for axis, name in enumerate(['x', 'y', 'z']):
    X = np.column_stack([err[:, axis], err_dot[:, axis]])
    Y = mu3[:, axis]
    params, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    kp_acc, kd_acc = params
    print(f"Axis {name}: kp_acc = {kp_acc:.6f}, kd_acc = {kd_acc:.6f}, Residual sum of squares: {residuals[0] if len(residuals) else 'N/A'}")
