# PX4 Offboard Control System Documentation

## Overview
This document explains the hierarchical control architecture implemented in the PX4 offboard control system, consisting of outer loop (position control) and inner loop (attitude/rate control) with Nonlinear Dynamic Inversion (NDI).

## Mathematical Model Reference

The control system is based on the following mathematical equations from the attached image:

### Position Dynamics (Equation 16)
```
⎛ṗn⎞   ⎛cθcψ    sφsθcψ - cφsψ    cφsθcψ + sφsψ⎞ ⎛u⎞
⎜ṗe⎟ = ⎜cθsψ    sφsθsψ + cφcψ    cφsθsψ - sφcψ⎟ ⎜v⎟
⎝ḣ ⎠   ⎝sθ           -sφcθ              -cφcθ   ⎠ ⎝w⎠
```

### Velocity Dynamics (Equation 17)
```
⎛u̇⎞   ⎛rv - qw⎞   ⎛-g sin θ    ⎞   1 ⎛ 0 ⎞
⎜v̇⎟ = ⎜pw - ru⎟ + ⎜g cos θ sin φ⎟ + — ⎜ 0 ⎟
⎝ẇ⎠   ⎝qu - pv⎠   ⎝g cos θ cos φ⎠   m ⎝-F⎠
```

### Attitude Kinematics (Equation 18)
```
⎛φ̇⎞   ⎛1  sin φ tan θ   cos φ tan θ⎞ ⎛p⎞
⎜θ̇⎟ = ⎜0     cos φ        -sin φ   ⎟ ⎜q⎟
⎝ψ̇⎠   ⎝0    sin φ/cos θ   cos φ/cos θ⎠ ⎝r⎠
```

### Angular Rate Dynamics (Equation 19)
```
⎛ṗ⎞   ⎛(Iy-Iz)/Ix qr⎞   ⎛1/Ix τφ⎞
⎜q̇⎟ = ⎜(Iz-Ix)/Iy pr⎟ + ⎜1/Iy τθ⎟
⎝ṙ⎠   ⎝(Ix-Iy)/Iz pq⎠   ⎝1/Iz τψ⎠
```

## Coordinate Systems and Conventions

### NED (North-East-Down) Frame
- **North (X)**: Positive direction points towards magnetic north
- **East (Y)**: Positive direction points towards east  
- **Down (Z)**: Positive direction points downward (towards Earth center)

### Body Frame
- **X-axis**: Forward direction of the vehicle
- **Y-axis**: Right side of the vehicle (starboard)
- **Z-axis**: Downward through the vehicle

### NED Conversions in the Code

#### 1. Position Control (Outer Loop)
```cpp
// Target position in NED frame
Eigen::Vector3d pos_design_{0.0, 0.0, -5.0};  // 5 meters altitude (negative Z in NED)

// Current position from PX4 (already in NED)
Eigen::Vector3d position(current_position_->x, current_position_->y, current_position_->z);
```

#### 2. Acceleration Command Transformation
The key NED conversion happens when transforming desired acceleration from world frame to body frame:

```cpp
// Yaw-only rotation matrix (from NED world to body frame)
Eigen::Matrix2d R_yaw;
R_yaw << cos(-yaw_predicted), -sin(-yaw_predicted),
         sin(-yaw_predicted),  cos(-yaw_predicted);

// Transform XY acceleration from world NED to body frame
Eigen::Vector2d a_des_body_xy = R_yaw * a_des_world_xy;
```

#### 3. Gravity Vector in NED
```cpp
Eigen::Vector3d g_vect_{0.0, 0.0, 9.81};  // Gravity points down (+Z in NED)
```

## Control System Architecture

## Outer Loop: Position Control

### Purpose
Controls the vehicle's position in 3D space by generating desired acceleration commands.

### Key Variables

#### Position Control Gains
```cpp
// XY (horizontal) position control gains
Eigen::Vector2d Kp_pos_xy_{0.55, 0.55};    // Proportional gains [North, East]
Eigen::Vector2d Kd_pos_xy_{1.10, 1.10};    // Derivative gains [North, East]  
Eigen::Vector2d Ki_pos_xy_{0.03, 0.03};    // Integral gains [North, East]

// Z (vertical) position control gains
double Kp_pos_z_ = 20.0;   // Z Position proportional gain
double Kd_pos_z_ = 20.0;   // Z Position derivative gain  
double Ki_pos_z_ = 1.0;    // Z Position integral gain
```

#### Position State Variables
```cpp
Eigen::Vector3d pos_design_{0.0, 0.0, -5.0};  // Target position [m] (NED frame)
Eigen::Vector2d pos_err_int_xy_{0.0, 0.0};   // XY position error integral
double pos_err_int_z_ = 0.0;                 // Z position error integral
Eigen::Vector3d a_des_;                      // Desired acceleration [m/s²] (NED frame)
```

#### Velocity Filtering
```cpp
Eigen::Vector2d vel_xy_filt_{0.0, 0.0};     // Filtered XY velocity [m/s]
double vel_lpf_alpha_{0.15};                 // Low-pass filter coefficient
```

#### Acceleration Limits
```cpp
double max_horz_acc_ = 3.0;   // Maximum horizontal acceleration [m/s²]
double max_vert_acc_ = g_;    // Maximum vertical acceleration [m/s²]
double max_xy_acc_step_{0.6}; // Maximum acceleration step (jerk limiting)
```

### Control Algorithm

#### XY Position Control (Horizontal)
1. **Error Calculation**: `pos_err_xy = pos_des_xy - pos_xy`
2. **Velocity Filtering**: Low-pass filter applied to reduce noise
3. **PID Control**: 
   ```
   a_des_world_xy = Kp * pos_err_xy - Kd * vel_xy_filt + Ki * pos_err_int_xy
   ```
4. **Saturation & Anti-windup**: Prevents integrator windup when saturated
5. **Frame Transformation**: Rotate acceleration command from world NED to body frame
6. **Jerk Limiting**: Limits rate of acceleration change

#### Z Position Control (Vertical)
1. **Error Calculation**: `pos_err_z = pos_des_z - pos_z`
2. **PID Control**: 
   ```
   a_des_z = Kp_pos_z * pos_err_z - Kd_pos_z * vel_z + Ki_pos_z * pos_err_int_z
   ```
3. **Sign Convention**: Negative sign applied to match NED frame convention
4. **Saturation**: Limited by `max_vert_acc_`

## Inner Loop: Attitude and Rate Control

### Purpose
Controls vehicle attitude and angular rates using Nonlinear Dynamic Inversion (NDI) to track desired acceleration commands from the outer loop.

### Key Variables

#### Attitude Control Gains
```cpp
double Kp_att_{4.00};  // Attitude proportional gain
double Kd_att_{0.0};   // Attitude derivative gain (rate damping)
```

#### NDI Rate Control Gains
```cpp
double Kp_ndi_rate_{0.100};  // NDI feedforward rate gain
double Kd_ndi_rate_{0.000};  // NDI feedforward rate derivative gain
```

#### Feedback Correction Gains
```cpp
double Kp_fb_rate_{0.180};  // Feedback rate proportional gain
double Kd_fb_rate_{0.010};  // Feedback rate derivative gain
```

#### Vehicle Parameters
```cpp
double mass_ = 2.0;  // Vehicle mass [kg]
double Ix_ = 0.022, Iy_ = 0.022, Iz_ = 0.04;  // Moments of inertia [kg⋅m²]
Eigen::Matrix3d Inertia_;  // Inertia matrix
```

#### Attitude State Variables
```cpp
Eigen::Vector3d desired_body_rates_{0.0, 0.0, 0.0}; // Desired [p,q,r] [rad/s]
Eigen::Vector3d mu3_cmd_{0.0, 0.0, 0.0};            // NDI torque command [N⋅m]
double max_tilt_angle_ = 0.7;                        // Maximum tilt [rad]
```

#### Rate Filtering
```cpp
Eigen::Vector3d omega_filt_{0.0, 0.0, 0.0};  // Filtered angular rates [rad/s]
double omega_lpf_alpha_{0.1};                 // Angular rate filter coefficient
```

### Control Algorithm

#### Attitude Control
1. **Desired Attitude**: Currently set to level flight (roll_des = pitch_des = 0)
2. **Attitude Error**: 
   ```
   roll_err = roll_des - roll
   pitch_err = pitch_des - pitch
   ```
3. **Rate Command Generation**:
   ```
   desired_body_rates(0) = Kp_att * roll_err - Kd_att * p
   desired_body_rates(1) = Kp_att * pitch_err - Kd_att * q
   desired_body_rates(2) = r  // Yaw rate passthrough
   ```

#### NDI Rate Control
1. **Rate Error**: `rate_error = desired_body_rates - omega_b`
2. **Rate Error Derivative**: Computed using finite differences
3. **NDI Feedforward**:
   ```
   omega_dot_design = Kp_ndi_rate * rate_error + Kd_ndi_rate * rate_error_dot
   ```
4. **NDI Torque Calculation**:
   ```
   mu3 = Inertia * omega_dot_design + omega_b × (Inertia * omega_b)
   ```

#### Feedback Correction
1. **Rate Error Computation**: `rate_err_vec = desired_body_rates - omega_b`
2. **Rate Error Derivative**: Estimated using finite differences
3. **Feedback Torques**:
   ```
   tau_fb = Kp_fb_rate * rate_err + Kd_fb_rate * rate_err_dot
   ```
4. **Combined Command**: `tau_cmd = mu3 + tau_fb`

## Advanced Features

### Yaw Feedforward Prediction
```cpp
double yaw_ff_dt_{0.035};          // Prediction horizon [s]
double yaw_rate_lpf_alpha_{0.2};   // Yaw rate filter coefficient
```
Predicts future yaw angle to compensate for control delays.

### Spin Mode
```cpp
bool spin_mode_{false};
double spin_torque_cmd_{2.0}; // Spin torque [N⋅m]
```
Enables controlled spinning maneuvers by commanding direct yaw torque.

### Integral Anti-windup
Prevents integrator windup when control commands saturate:
```cpp
// Back out integration step if saturated
if (sat_x) pos_err_int_xy_(0) -= dt * pos_err_xy(0);
if (sat_y) pos_err_int_xy_(1) -= dt * pos_err_xy(1);
```

## Control Flow Summary

1. **Outer Loop** (Position Control):
   - Reads current position and velocity (NED frame)
   - Computes position errors
   - Applies PID control with filtering and limiting
   - Generates desired acceleration commands
   - Transforms to body frame for inner loop

2. **Inner Loop** (Attitude/Rate Control):
   - Converts desired acceleration to desired attitude
   - Computes attitude errors
   - Generates desired body rates
   - Uses NDI to compute feedforward torques
   - Adds feedback correction torques
   - Publishes thrust and torque commands

3. **Frame Transformations**:
   - Position control operates in NED world frame
   - Acceleration commands transformed to body frame
   - Attitude control operates in body frame
   - All outputs sent to PX4 in appropriate frames

## Coordinate Frame Summary

| Variable | Frame | Units | Description |
|----------|-------|-------|-------------|
| `pos_design_` | NED World | [m] | Target position |
| `current_position_` | NED World | [m] | Current position |
| `a_des_world_xy` | NED World | [m/s²] | Desired acceleration (world) |
| `a_des_body_xy` | Body | [m/s²] | Desired acceleration (body) |
| `roll, pitch, yaw` | Euler | [rad] | Vehicle attitude |
| `omega_b` | Body | [rad/s] | Angular rates |
| `mu3_cmd_` | Body | [N⋅m] | Torque commands |

This hierarchical control structure ensures stable position tracking while maintaining good disturbance rejection and handling the nonlinear dynamics of the multirotor vehicle.