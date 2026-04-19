#!/usr/bin/env python3
"""
Drone Flight Visualization Script (fixed transforms)
Reads flight log CSV data and creates an animated 3D visualization of the drone's trajectory
with a cross-shaped body representing the drone chassis and motors.

Fixes applied:
 - Converts PX4 NED (pos_x=North, pos_y=East, pos_z=Down) -> ENU (x=East, y=North, z=Up) for plotting.
 - Converts attitude/rotation (body->NED) -> body->ENU so body points and torque arrows are correct.
 - Computes tilt angles from the rotated body z-axis for accurate north/east tilt.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import cm
from matplotlib.colors import Normalize
import sys
import os

class Arrow3D(FancyArrowPatch):
    """Custom 3D arrow for visualization"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def rotation_matrix(roll, pitch, yaw):
    """
    Create a rotation matrix from Euler angles (roll, pitch, yaw) using ZYX intrinsic (3-2-1) convention.
    This constructs R_body_to_NED when given PX4's roll/pitch/yaw Euler angles.
    
    PX4 Convention: Body frame rotates relative to NED frame via:
    1. Yaw rotation around body Z (down) axis
    2. Pitch rotation around intermediate Y (right) axis  
    3. Roll rotation around final X (forward) axis
    
    The rotation matrix transforms vectors from body frame to NED frame: v_NED = R @ v_body
    """
    # Roll (rotation around X-axis) - right wing down is positive
    cr = np.cos(roll)
    sr = np.sin(roll)
    R_x = np.array([
        [1,  0,   0 ],
        [0,  cr, -sr],
        [0,  sr,  cr]
    ])
    
    # Pitch (rotation around Y-axis) - nose up is positive
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    R_y = np.array([
        [ cp, 0,  sp],
        [ 0,  1,  0 ],
        [-sp, 0,  cp]
    ])
    
    # Yaw (rotation around Z-axis) - clockwise from above (NED) is positive
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    R_z = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [0,   0,  1]
    ])
    
    # Combined rotation: R_body_to_NED = R_z(yaw) * R_y(pitch) * R_x(roll)
    # Applied right-to-left: first roll, then pitch, then yaw
    R = R_z @ R_y @ R_x
    return R


def create_drone_body(arm_length=0.3):
    """
    Create the drone body as a cross shape with 4 arms
    Returns vertices for the drone structure based on motor positions in the body frame.
    Body frame convention assumed (PX4): X = forward (North), Y = right (East), Z = down.
    Motor positions arranged accordingly (units: meters).
    """
    # Center of the drone (body frame)
    center = np.array([0.0, 0.0, 0.0])
    
    # Motor coordinates in body frame (X forward, Y right, Z down)
    motor1 = np.array([0.13,  0.22, 0.0])   # Front-right (forward, right)
    motor2 = np.array([-0.13, -0.20, 0.0])  # Rear-left
    motor3 = np.array([0.13, -0.22, 0.0])   # Rear-right
    motor4 = np.array([-0.13,  0.20, 0.0])  # Front-left
    
    return center, motor1, motor2, motor3, motor4


def transform_drone(position_enu, roll, pitch, yaw, arm_length=0.3):
    """
    Transform drone body points from body frame to world (ENU) frame.

    Process:
    1. roll/pitch/yaw are PX4 Euler angles describing body orientation relative to NED frame
    2. Compute R_body_to_NED using rotation_matrix()
    3. Convert to R_body_to_ENU via coordinate transformation
    
    Coordinate transformation NED->ENU:
    - NED: X=North, Y=East, Z=Down
    - ENU: X=East, Y=North, Z=Up
    - Mapping: ENU_x = NED_y, ENU_y = NED_x, ENU_z = -NED_z
    
    For rotation matrices:
    v_ENU = R_NED_to_ENU @ v_NED = R_NED_to_ENU @ R_body_to_NED @ v_body
    So: R_body_to_ENU = R_NED_to_ENU @ R_body_to_NED
    """
    # Get rotation matrix: body -> NED
    R_body_to_NED = rotation_matrix(roll, pitch, yaw)

    # Rotation matrix from NED to ENU frame
    R_NED_to_ENU = np.array([[0.0, 1.0,  0.0],   # ENU_x = NED_y (East)
                              [1.0, 0.0,  0.0],   # ENU_y = NED_x (North)
                              [0.0, 0.0, -1.0]])  # ENU_z = -NED_z (Up)
    
    # Compose: body -> NED -> ENU
    R_body_to_ENU = R_NED_to_ENU @ R_body_to_NED

    # Get body frame points (defined in body frame: X=forward/North, Y=right/East, Z=down)
    center_b, motor1_b, motor2_b, motor3_b, motor4_b = create_drone_body(arm_length)

    # Transform body points to ENU world frame
    center_world = position_enu
    motor1_world = position_enu + R_body_to_ENU @ motor1_b
    motor2_world = position_enu + R_body_to_ENU @ motor2_b
    motor3_world = position_enu + R_body_to_ENU @ motor3_b
    motor4_world = position_enu + R_body_to_ENU @ motor4_b
    
    return center_world, motor1_world, motor2_world, motor3_world, motor4_world, R_body_to_ENU


class DroneVisualizer:
    def __init__(self, csv_file, skip_frames=10, arm_length=0.3, tilt_exaggeration=3.0):
        """
        Initialize the drone visualizer
        """
        self.csv_file = csv_file
        self.skip_frames = skip_frames
        self.arm_length = arm_length
        self.tilt_exaggeration = tilt_exaggeration
        
        # Load data
        print(f"Loading flight data from: {csv_file}")
        self.data = pd.read_csv(csv_file)
        
        # Downsample data for animation
        self.data = self.data.iloc[::skip_frames].reset_index(drop=True)
        print(f"Loaded {len(self.data)} frames (downsampled by {skip_frames})")
        
        # Extract relevant columns (PX4 logs are NED: x=North, y=East, z=Down)
        pos_ned = self.data[['pos_x', 'pos_y', 'pos_z']].values  # [N, E, D]
        # Convert NED -> ENU and Z-up for plotting: ENU_x = East, ENU_y = North, ENU_z = -Down
        self.positions = np.column_stack([pos_ned[:, 1], pos_ned[:, 0], -pos_ned[:, 2]])
        
        self.roll = self.data['roll'].values   # radians, PX4 body→NED convention
        self.pitch = self.data['pitch'].values
        self.yaw = self.data['yaw'].values
        self.timestamps = self.data['timestamp'].values
        
        # Load commanded torques from CSV (if present) - these are in body frame
        if all(col in self.data.columns for col in ['torque_x', 'torque_y', 'torque_z']):
            self.torque_cmd_x = self.data['torque_x'].values  # Commanded roll torque (body frame)
            self.torque_cmd_y = self.data['torque_y'].values  # Commanded pitch torque (body frame)
            self.torque_cmd_z = self.data['torque_z'].values  # Commanded yaw torque (body frame)
            print("Loaded commanded torques from CSV")
        else:
            self.torque_cmd_x = np.zeros(len(self.data))
            self.torque_cmd_y = np.zeros(len(self.data))
            self.torque_cmd_z = np.zeros(len(self.data))
            print("Warning: No commanded torque columns found in CSV")
        
        # Compute actual torques from motor values (if present)
        if all(col in self.data.columns for col in ['motor1', 'motor2', 'motor3', 'motor4']):
            # Raw motor commands
            m1_raw = self.data['motor1'].values
            m2_raw = self.data['motor2'].values
            m3_raw = self.data['motor3'].values
            m4_raw = self.data['motor4'].values
            
            # # Square motor values (thrust proportional to RPM^2)
            m1 = m1_raw 
            m2 = m2_raw 
            m3 = m3_raw 
            m4 = m4_raw 
            
            # Motor positions in body frame (X forward/North, Y right/East, Z down)
            # M1: front-right  [0.13,  0.22, 0]
            # M2: rear-left    [-0.13, -0.20, 0]
            # M3: rear-right   [0.13, -0.22, 0]
            # M4: front-left   [-0.13,  0.20, 0]
            
            self.torque_motor_x = (m1 * 0.22 + m3 * (-0.22)) + (m2 * (-0.20) + m4 * 0.20)
            self.torque_motor_y = -(m2 * (-0.13) + m3 * 0.13) - (m1 * 0.13 + m4 * (-0.13))
            self.torque_motor_z = (m1 + m2 + m3 + m4) * 0.1  # All CCW motors
            
            # Normalize motor torques to have unit norm (scale to match commanded torque magnitude)
            # motor_torque_vector = np.column_stack([self.torque_motor_x, self.torque_motor_y, self.torque_motor_z])
            # motor_torque_norm = np.linalg.norm(motor_torque_vector, axis=1)
            
            # # Compute commanded torque norm
            # cmd_torque_vector = np.column_stack([self.torque_cmd_x, self.torque_cmd_y, self.torque_cmd_z])
            # cmd_torque_norm = np.linalg.norm(cmd_torque_vector, axis=1)
            
            # # Scale motor torques to match commanded torque magnitude (where non-zero)
            # scale_factor = np.ones(len(self.data))
            # nonzero_mask = motor_torque_norm > 1e-10
            # if np.any(nonzero_mask):
            #     # Compute average scale ratio for non-zero entries
            #     avg_cmd_norm = np.mean(cmd_torque_norm[nonzero_mask])
            #     avg_motor_norm = np.mean(motor_torque_norm[nonzero_mask])
            #     if avg_motor_norm > 0:
            #         scale_factor[:] = avg_cmd_norm / avg_motor_norm
            
            # Apply scaling
            # self.torque_motor_x *= scale_factor
            # self.torque_motor_y *= scale_factor
            # self.torque_motor_z *= scale_factor
            
            # print("Computed torques from motor values (squared and normalized)")
            # print(f"Scale factor applied: {scale_factor[0]:.6f}")
            # # Print comparison statistics
            # print(f"Commanded torque_x range: [{self.torque_cmd_x.min():.6f}, {self.torque_cmd_x.max():.6f}]")
            # print(f"Motor torque_x range:     [{self.torque_motor_x.min():.6f}, {self.torque_motor_x.max():.6f}]")
            # print(f"Commanded torque_y range: [{self.torque_cmd_y.min():.6f}, {self.torque_cmd_y.max():.6f}]")
            # print(f"Motor torque_y range:     [{self.torque_motor_y.min():.6f}, {self.torque_motor_y.max():.6f}]")
        else:
            self.torque_motor_x = np.zeros(len(self.data))
            self.torque_motor_y = np.zeros(len(self.data))
            self.torque_motor_z = np.zeros(len(self.data))
        
        # motor thrusts (if present) for coloring
        self.motor1_thrust = self.data['motor1'].values if 'motor1' in self.data.columns else np.zeros(len(self.data))
        self.motor2_thrust = self.data['motor2'].values if 'motor2' in self.data.columns else np.zeros(len(self.data))
        self.motor3_thrust = self.data['motor3'].values if 'motor3' in self.data.columns else np.zeros(len(self.data))
        self.motor4_thrust = self.data['motor4'].values if 'motor4' in self.data.columns else np.zeros(len(self.data))
        motor_array = np.column_stack([self.motor1_thrust, self.motor2_thrust, self.motor3_thrust, self.motor4_thrust])
        self.instant_avg_thrust = np.mean(motor_array, axis=1)
        
        # colormap
        try:
            self.cmap = plt.get_cmap('RdYlGn_r')
        except:
            self.cmap = cm.get_cmap('RdYlGn_r')
        
        # Figure
        self.fig = plt.figure(figsize=(16, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Precompute tilts and world torques in ENU (so axes are consistent)
        self._precompute_tilts_and_torques()
        
        # Initialize plot
        self.init_plot()
        
    def _precompute_tilts_and_torques(self):
        """
        For each timestep compute:
         - ENU tilt angles (east tilt, north tilt) derived from the body z-axis in ENU
         - Commanded torques transformed into ENU world frame
         - Motor-computed torques transformed into ENU world frame
        """
        n = len(self.data)
        self.ned_roll = np.zeros(n)   # east tilt
        self.ned_pitch = np.zeros(n)  # north tilt
        
        # ENU torques - commanded
        self.enu_torque_cmd_x = np.zeros(n)
        self.enu_torque_cmd_y = np.zeros(n)
        self.enu_torque_cmd_z = np.zeros(n)
        
        # ENU torques - computed from motors
        self.enu_torque_motor_x = np.zeros(n)
        self.enu_torque_motor_y = np.zeros(n)
        self.enu_torque_motor_z = np.zeros(n)
        
        # Rotation matrix from NED to ENU frame (as used in transform_drone)
        R_NED_to_ENU = np.array([[0.0, 1.0,  0.0],   # ENU_x = NED_y (East)
                                  [1.0, 0.0,  0.0],   # ENU_y = NED_x (North)
                                  [0.0, 0.0, -1.0]])  # ENU_z = -NED_z (Up)
        
        for i in range(n):
            r = self.roll[i]
            p = self.pitch[i]
            y = self.yaw[i]
            
            # Compute R_body_to_NED from PX4 Euler angles
            R_body_to_NED = rotation_matrix(r, p, y)
            
            # Transform to R_body_to_ENU
            R_body_to_ENU = R_NED_to_ENU @ R_body_to_NED
            
            # Body z-axis in ENU frame (body frame: Z points down when level)
            # Transform body Z-axis unit vector [0, 0, 1] to ENU frame
            body_z_in_enu = R_body_to_ENU @ np.array([0.0, 0.0, 1.0])
            
            # Compute tilt angles from the body Z-axis direction in ENU
            # When level, body_z_in_enu should be [0, 0, -1] (pointing down in ENU = up in body)
            # Tilt angles measure deviation from vertical (ENU Z-axis)
            # East tilt: rotation about North axis (Y in ENU) - positive = tilted East
            # North tilt: rotation about East axis (X in ENU) - positive = tilted North
            # Use atan2 for proper quadrant handling
            self.ned_roll[i] = np.arctan2(body_z_in_enu[0], -body_z_in_enu[2])  # East tilt
            self.ned_pitch[i] = np.arctan2(body_z_in_enu[1], -body_z_in_enu[2]) # North tilt
            
            # Transform commanded body torques to ENU world frame
            # Body torques: X=roll (about forward axis), Y=pitch (about right axis), Z=yaw (about down axis)
            body_tau_cmd = np.array([self.torque_cmd_x[i], self.torque_cmd_y[i], self.torque_cmd_z[i]])
            tau_cmd_enu = R_body_to_ENU @ body_tau_cmd
            self.enu_torque_cmd_x[i] = tau_cmd_enu[0]
            self.enu_torque_cmd_y[i] = tau_cmd_enu[1]
            self.enu_torque_cmd_z[i] = tau_cmd_enu[2]
            
            # Transform motor-computed body torques to ENU world frame
            body_tau_motor = np.array([self.torque_motor_x[i], self.torque_motor_y[i], self.torque_motor_z[i]])
            tau_motor_enu = R_body_to_ENU @ body_tau_motor
            self.enu_torque_motor_x[i] = tau_motor_enu[0]
            self.enu_torque_motor_y[i] = tau_motor_enu[1]
            self.enu_torque_motor_z[i] = tau_motor_enu[2]
    
    def init_plot(self):
        """Initialize the 3D plot"""
        self.fig.clear()
        gs = self.fig.add_gridspec(3, 1, height_ratios=[2, 1.5, 1.5], hspace=0.4)
        self.ax = self.fig.add_subplot(gs[0], projection='3d')  # 3D view (top)
        self.ax_east = self.fig.add_subplot(gs[1])  # East tilt & torque
        self.ax_north = self.fig.add_subplot(gs[2])  # North tilt & torque

        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.set_zlabel('Up (m)')
        
        # Axis limits
        margin = 1.0
        x_min, x_max = self.positions[:, 0].min() - margin, self.positions[:, 0].max() + margin
        y_min, y_max = self.positions[:, 1].min() - margin, self.positions[:, 1].max() + margin
        z_min, z_max = self.positions[:, 2].min() - margin, self.positions[:, 2].max() + margin
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)
        
        # Trajectory
        self.trajectory_line, = self.ax.plot([], [], [], 'b-', alpha=0.3, linewidth=1, label='Trajectory')
        
        # Drone body arms
        self.drone_arm1, = self.ax.plot([], [], [], 'k-', linewidth=3, label='Drone Frame')
        self.drone_arm2, = self.ax.plot([], [], [], 'k-', linewidth=3)
        self.drone_arm3, = self.ax.plot([], [], [], 'k-', linewidth=2, alpha=0.5)
        self.drone_arm4, = self.ax.plot([], [], [], 'k-', linewidth=2, alpha=0.5)
        
        # Motor markers
        self.motor1_scatter = self.ax.scatter([], [], [], s=150, marker='o', edgecolors='black', linewidths=2)
        self.motor2_scatter = self.ax.scatter([], [], [], s=150, marker='o', edgecolors='black', linewidths=2)
        self.motor3_scatter = self.ax.scatter([], [], [], s=150, marker='o', edgecolors='black', linewidths=2)
        self.motor4_scatter = self.ax.scatter([], [], [], s=150, marker='o', edgecolors='black', linewidths=2)
        
        # Current position
        self.position_marker = self.ax.scatter([], [], [], c='cyan', s=50, marker='o', label='Current Position')
        
        # Torque arrows
        self.torque_arrow_x = None
        self.torque_arrow_y = None
        self.torque_arrow_z = None
        
        # Time text
        self.time_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes)
        
        self.ax.legend()
        
        # East tilt subplot
        self.ax_east.set_xlabel('Time (s)', fontsize=9)
        self.ax_east.set_ylabel('Angle (rad)', fontsize=9)
        self.ax_east.set_title('East Tilt, Commanded & Motor Torque', fontsize=10)
        self.ax_east.grid(True, alpha=0.3)
        self.ax_east.tick_params(labelsize=8)
        self.ax_east_torque = self.ax_east.twinx()
        self.ax_east_torque.set_ylabel('Torque', fontsize=9, color='red')
        self.ax_east_torque.tick_params(labelsize=8, colors='red')
        self.east_tilt_line, = self.ax_east.plot([], [], 'b-', label='East Tilt', linewidth=2)
        self.east_torque_cmd_line, = self.ax_east_torque.plot([], [], 'r-', label='Cmd Torque', linewidth=2, alpha=0.8)
        self.east_torque_motor_line, = self.ax_east_torque.plot([], [], 'orange', label='Motor Torque', linewidth=1.5, alpha=0.6, linestyle='--')
        self.east_vline = self.ax_east.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        self.ax_east.set_xlim(self.timestamps[0], self.timestamps[-1])
        east_tilt_max = max(abs(self.ned_roll.min()), abs(self.ned_roll.max()))
        if east_tilt_max > 0:
            self.ax_east.set_ylim(-east_tilt_max * 1.2, east_tilt_max * 1.2)
        else:
            self.ax_east.set_ylim(-0.1, 0.1)
        east_torque_max = max(np.abs(self.enu_torque_cmd_x).max(), np.abs(self.enu_torque_motor_x).max())
        if east_torque_max > 0:
            self.ax_east_torque.set_ylim(-east_torque_max * 0.8, east_torque_max * 0.8)
        else:
            self.ax_east_torque.set_ylim(-1, 1)
        lines1, labels1 = self.ax_east.get_legend_handles_labels()
        lines2, labels2 = self.ax_east_torque.get_legend_handles_labels()
        self.ax_east.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
        # North tilt subplot
        self.ax_north.set_xlabel('Time (s)', fontsize=9)
        self.ax_north.set_ylabel('Angle (rad)', fontsize=9)
        self.ax_north.set_title('North Tilt, Commanded & Motor Torque', fontsize=10)
        self.ax_north.grid(True, alpha=0.3)
        self.ax_north.tick_params(labelsize=8)
        self.ax_north_torque = self.ax_north.twinx()
        self.ax_north_torque.set_ylabel('Torque', fontsize=9, color='red')
        self.ax_north_torque.tick_params(labelsize=8, colors='red')
        self.north_tilt_line, = self.ax_north.plot([], [], 'g-', label='North Tilt', linewidth=2)
        self.north_torque_cmd_line, = self.ax_north_torque.plot([], [], 'r-', label='Cmd Torque', linewidth=2, alpha=0.8)
        self.north_torque_motor_line, = self.ax_north_torque.plot([], [], 'orange', label='Motor Torque', linewidth=1.5, alpha=0.6, linestyle='--')
        self.north_vline = self.ax_north.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        self.ax_north.set_xlim(self.timestamps[0], self.timestamps[-1])
        north_tilt_max = max(abs(self.ned_pitch.min()), abs(self.ned_pitch.max()))
        if north_tilt_max > 0:
            self.ax_north.set_ylim(-north_tilt_max * 1.2, north_tilt_max * 1.2)
        else:
            self.ax_north.set_ylim(-0.1, 0.1)
        north_torque_max = max(np.abs(self.enu_torque_cmd_y).max(), np.abs(self.enu_torque_motor_y).max())
        if north_torque_max > 0:
            self.ax_north_torque.set_ylim(-north_torque_max * 0.8, north_torque_max * 0.8)
        else:
            self.ax_north_torque.set_ylim(-1, 1)
        lines1, labels1 = self.ax_north.get_legend_handles_labels()
        lines2, labels2 = self.ax_north_torque.get_legend_handles_labels()
        self.ax_north.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
        
    def update(self, frame):
        """Update function for animation"""
        if frame >= len(self.positions):
            return
        
        # Current state (positions already in ENU: x=East,y=North,z=Up)
        pos = self.positions[frame]
        # Exaggerate roll/pitch visualization using the precomputed tilt (more stable)
        roll_vis = self.ned_roll[frame] * self.tilt_exaggeration
        pitch_vis = self.ned_pitch[frame] * self.tilt_exaggeration
        # For rotating the body we still use raw roll/pitch/yaw but convert during transform
        roll = self.roll[frame]
        pitch = self.pitch[frame]
        yaw = self.yaw[frame]
        time = self.timestamps[frame]
        
        # motor thrusts
        instant_avg = self.instant_avg_thrust[frame]
        m1_thrust_abs = self.motor1_thrust[frame]
        m2_thrust_abs = self.motor2_thrust[frame]
        m3_thrust_abs = self.motor3_thrust[frame]
        m4_thrust_abs = self.motor4_thrust[frame]
        
        if instant_avg > 0:
            m1_deviation = m1_thrust_abs - instant_avg
            m2_deviation = m2_thrust_abs - instant_avg
            m3_deviation = m3_thrust_abs - instant_avg
            m4_deviation = m4_thrust_abs - instant_avg
            m1_thrust = m1_thrust_abs / instant_avg
            m2_thrust = m2_thrust_abs / instant_avg
            m3_thrust = m3_thrust_abs / instant_avg
            m4_thrust = m4_thrust_abs / instant_avg
        else:
            m1_deviation = m2_deviation = m3_deviation = m4_deviation = 0.0
            m1_thrust = m2_thrust = m3_thrust = m4_thrust = 1.0
        
        deviations = np.array([m1_deviation, m2_deviation, m3_deviation, m4_deviation])
        max_abs_deviation = np.max(np.abs(deviations))
        if max_abs_deviation < 0.01:
            max_abs_deviation = 0.01
        dynamic_norm = Normalize(vmin=-max_abs_deviation, vmax=max_abs_deviation)
        motor1_color = self.cmap(dynamic_norm(m1_deviation))
        motor2_color = self.cmap(dynamic_norm(m2_deviation))
        motor3_color = self.cmap(dynamic_norm(m3_deviation))
        motor4_color = self.cmap(dynamic_norm(m4_deviation))
        
        # Update trajectory
        self.trajectory_line.set_data(self.positions[:frame+1, 0], self.positions[:frame+1, 1])
        self.trajectory_line.set_3d_properties(self.positions[:frame+1, 2])
        
        # Transform drone body and get rotation used
        center, motor1, motor2, motor3, motor4, R_b2e = transform_drone(pos, roll, pitch, yaw, self.arm_length)
        
        # Drone arms update
        self.drone_arm1.set_data([motor1[0], motor2[0]], [motor1[1], motor2[1]])
        self.drone_arm1.set_3d_properties([motor1[2], motor2[2]])
        self.drone_arm2.set_data([motor3[0], motor4[0]], [motor3[1], motor4[1]])
        self.drone_arm2.set_3d_properties([motor3[2], motor4[2]])
        self.drone_arm3.set_data([motor1[0], motor3[0]], [motor1[1], motor3[1]])
        self.drone_arm3.set_3d_properties([motor1[2], motor3[2]])
        self.drone_arm4.set_data([motor2[0], motor4[0]], [motor2[1], motor4[1]])
        self.drone_arm4.set_3d_properties([motor2[2], motor4[2]])
        
        # Update motor scatter points and colors
        self.motor1_scatter._offsets3d = ([motor1[0]], [motor1[1]], [motor1[2]])
        self.motor1_scatter.set_color(motor1_color)
        self.motor2_scatter._offsets3d = ([motor2[0]], [motor2[1]], [motor2[2]])
        self.motor2_scatter.set_color(motor2_color)
        self.motor3_scatter._offsets3d = ([motor3[0]], [motor3[1]], [motor3[2]])
        self.motor3_scatter.set_color(motor3_color)
        self.motor4_scatter._offsets3d = ([motor4[0]], [motor4[1]], [motor4[2]])
        self.motor4_scatter.set_color(motor4_color)
        
        # Update position marker
        self.position_marker._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
        
        # Torques (transformed earlier into ENU) - use commanded for arrows
        torque_x_val = self.enu_torque_cmd_x[frame]
        torque_y_val = self.enu_torque_cmd_y[frame]
        torque_z_val = self.enu_torque_cmd_z[frame]
        
        # Remove old arrows
        if self.torque_arrow_x is not None:
            try:
                self.torque_arrow_x.remove()
            except (ValueError, AttributeError):
                pass
            self.torque_arrow_x = None
        if self.torque_arrow_y is not None:
            try:
                self.torque_arrow_y.remove()
            except (ValueError, AttributeError):
                pass
            self.torque_arrow_y = None
        if self.torque_arrow_z is not None:
            try:
                self.torque_arrow_z.remove()
            except (ValueError, AttributeError):
                pass
            self.torque_arrow_z = None
        
        # Add new torque arrows scaled
        arrow_scale = 0.3
        max_torque = max(np.abs(self.enu_torque_cmd_x).max(), np.abs(self.enu_torque_cmd_y).max(), np.abs(self.enu_torque_cmd_z).max())
        if max_torque > 0:
            tx = torque_x_val / max_torque * arrow_scale
            ty = torque_y_val / max_torque * arrow_scale
            tz = torque_z_val / max_torque * arrow_scale
            
            # East (x axis) torque - red arrow
            if abs(tx) > 0.01:
                self.torque_arrow_x = Arrow3D([pos[0], pos[0] + tx], 
                                             [pos[1], pos[1]], 
                                             [pos[2], pos[2]], 
                                             mutation_scale=20, lw=3, 
                                             arrowstyle='-|>', color='red', alpha=0.7)
                self.ax.add_artist(self.torque_arrow_x)
            # North (y axis) torque - green arrow
            if abs(ty) > 0.01:
                self.torque_arrow_y = Arrow3D([pos[0], pos[0]], 
                                             [pos[1], pos[1] + ty], 
                                             [pos[2], pos[2]], 
                                             mutation_scale=20, lw=3, 
                                             arrowstyle='-|>', color='green', alpha=0.7)
                self.ax.add_artist(self.torque_arrow_y)
            # Up (z axis) torque - blue arrow
            if abs(tz) > 0.01:
                self.torque_arrow_z = Arrow3D([pos[0], pos[0]], 
                                             [pos[1], pos[1]], 
                                             [pos[2], pos[2] + tz], 
                                             mutation_scale=20, lw=3, 
                                             arrowstyle='-|>', color='blue', alpha=0.7)
                self.ax.add_artist(self.torque_arrow_z)
        
        # Update East & North tilt plots
        self.east_tilt_line.set_data(self.timestamps[:frame+1], self.ned_roll[:frame+1])
        self.east_torque_cmd_line.set_data(self.timestamps[:frame+1], self.enu_torque_cmd_x[:frame+1])  # Commanded
        self.east_torque_motor_line.set_data(self.timestamps[:frame+1], self.enu_torque_motor_x[:frame+1])  # From motors
        self.east_vline.set_xdata([time, time])
        self.north_tilt_line.set_data(self.timestamps[:frame+1], self.ned_pitch[:frame+1])
        self.north_torque_cmd_line.set_data(self.timestamps[:frame+1], self.enu_torque_cmd_y[:frame+1])  # Commanded
        self.north_torque_motor_line.set_data(self.timestamps[:frame+1], self.enu_torque_motor_y[:frame+1])  # From motors
        self.north_vline.set_xdata([time, time])
        
        # Display numbers (use ENU values)
        east_tilt = self.ned_roll[frame]
        north_tilt = self.ned_pitch[frame]
        actual_yaw = self.yaw[frame]
        
        self.time_text.set_text(
            f'Time: {time:.2f} s | Frame: {frame}/{len(self.positions)} | Tilt Scale: {self.tilt_exaggeration}x\n'
            f'ENU Tilts - North: {north_tilt:+.3f} | East: {east_tilt:+.3f} | Yaw (rad): {actual_yaw:+.3f}\n'
            f'Cmd Torques - E: {self.enu_torque_cmd_x[frame]:+.3f} | N: {self.enu_torque_cmd_y[frame]:+.3f}\n'
            f'Motor Torques - E: {self.enu_torque_motor_x[frame]:+.3f} | N: {self.enu_torque_motor_y[frame]:+.3f}'
        )
        
        return (self.trajectory_line, self.drone_arm1, self.drone_arm2, self.drone_arm3, self.drone_arm4,
                self.motor1_scatter, self.motor2_scatter, self.motor3_scatter, self.motor4_scatter,
                self.position_marker, self.time_text,
                self.east_tilt_line, self.east_torque_cmd_line, self.east_torque_motor_line, self.east_vline,
                self.north_tilt_line, self.north_torque_cmd_line, self.north_torque_motor_line, self.north_vline)
    
    def animate(self, interval=20, save_file=None, start_time=0.0):
        """
        Create and display the animation
        """
        start_frame = 0
        for i, t in enumerate(self.timestamps):
            if t >= start_time:
                start_frame = i
                break
        
        print(f"Starting animation from {self.timestamps[start_frame]:.2f}s (frame {start_frame})")
        
        def frame_generator():
            for i in range(start_frame, len(self.positions)):
                yield i
        
        anim = FuncAnimation(self.fig, self.update, frames=frame_generator,
                           interval=interval, blit=False, repeat=False)
        
        if save_file:
            print(f"Saving animation to {save_file}...")
            if save_file.endswith('.gif'):
                anim.save(save_file, writer='pillow', fps=20)
            elif save_file.endswith('.mp4'):
                anim.save(save_file, writer='ffmpeg', fps=20)
            print("Animation saved!")
        
        plt.show()


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python visualize_flight.py <path_to_csv_file> [skip_frames] [arm_length] [tilt_exaggeration]")
        print("\nExample: python visualize_flight.py flight_data/logs/flight_log_20251014_133011.csv 50 0.3 3.0")
        print("\nSearching for CSV files in flight_data/logs/...")
        
        # Try to find CSV files
        log_dir = "flight_data/logs"
        if os.path.exists(log_dir):
            csv_files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
            if csv_files:
                print(f"\nFound {len(csv_files)} CSV file(s):")
                for i, f in enumerate(csv_files, 1):
                    print(f"  {i}. {f}")
                
                # Use the most recent file
                csv_files.sort(reverse=True)
                csv_file = os.path.join(log_dir, csv_files[0])
                print(f"\nUsing most recent file: {csv_file}")
            else:
                print("No CSV files found!")
                sys.exit(1)
        else:
            print(f"Directory {log_dir} not found!")
            sys.exit(1)
    else:
        csv_file = sys.argv[1]
    
    # Parse optional arguments
    skip_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    arm_length = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
    tilt_exaggeration = float(sys.argv[4]) if len(sys.argv) > 4 else 3.0
    
    # Create visualizer
    viz = DroneVisualizer(csv_file, skip_frames=skip_frames, arm_length=arm_length, tilt_exaggeration=tilt_exaggeration)
    
    # Optional: save animation
    save_file = None
    
    # Run animation (interval=20ms for faster playback, start at 10 seconds)
    viz.animate(interval=20, save_file=save_file, start_time=19.0)


if __name__ == "__main__":
    main()
