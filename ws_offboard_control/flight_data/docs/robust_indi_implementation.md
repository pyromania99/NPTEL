# Robust Hybrid NDI + INDI Control Implementation

## Overview
This document explains the implementation of the suggested robust "Hybrid NDI + guarded INDI" control improvements for spinning quadrotor control.

## Implemented Changes

### 1. **Torque Rotation Between Frames (Small-Angle Approximation)**
```cpp
// Previous torque rotation using small-angle approximation
Eigen::Vector3d delta_theta = omega_body * dt;
Eigen::Vector3d tau_prev_rot = tau_prev_body_ + delta_theta.cross(tau_prev_body_);
```
**Benefits:**
- More computationally efficient than full rotation matrices
- Physically consistent for fast spinning applications
- Accounts for angular velocity effects on torque commands

### 2. **Filtered Angular Rates Before Differentiation**
```cpp
// Filter omega before differentiation to reduce noise
omega_filt_ = (1.0 - alpha_omega_lp_) * omega_filt_ + alpha_omega_lp_ * omega_body;
Eigen::Vector3d omega_dot_raw = (omega_filt_ - omega_filt_prev_) / std::max(dt, 1e-6);
omega_dot_meas_filt_ = (1.0 - alpha_acc_lp_) * omega_dot_meas_filt_ + alpha_acc_lp_ * omega_dot_raw;
```
**Benefits:**
- Reduces noise in angular acceleration estimation
- Two-stage filtering: omega rates → angular acceleration
- Configurable filter parameters (alpha_omega_lp_, alpha_acc_lp_)

### 3. **Yaw-Rate Scaling on INDI Increment**
```cpp
// Scale down INDI at high yaw rates for stability
double yaw_rate = std::abs(omega_body(2));
double scale = (yaw_rate > 5.0) ? 1.0 / (1.0 + 0.5 * (yaw_rate - 5.0)) : 1.0;
```
**Benefits:**
- Prevents INDI instability during aggressive yaw maneuvers
- Threshold at 5 rad/s (~286 deg/s)
- Graceful degradation rather than hard switching

### 4. **Delta Tau Clamping**
```cpp
// Limit INDI increments to prevent large control jumps
for (int i = 0; i < 3; ++i) {
    delta_tau(i) = std::clamp(delta_tau(i), -3.0, 3.0);
}
```
**Benefits:**
- Prevents excessive control increments
- Maintains system stability
- 3.0 N⋅m limit chosen based on typical quadrotor torque ranges

### 5. **Moderate Blend Ratio (0.3–0.5)**
```cpp
// Conservative blending between NDI and INDI
tau_cmd = (1.0 - indi_blend_ratio_) * mu3_cmd_ + indi_blend_ratio_ * tau_indi;
```
**Benefits:**
- Default blend_ratio = 0.4 (configurable via JSON)
- 60% NDI (model-based) + 40% INDI (incremental)
- Maintains baseline NDI robustness while adding INDI benefits

## Configuration Parameters

### New JSON Parameters Added:
```json
"indi": {
    "enabled": true,
    "scale": 0.8168886881742098,
    "blend_ratio": 0.4,
    "alpha_omega_lp": 0.15,
    "alpha_acc_lp": 0.25
}
```

**Parameter Descriptions:**
- `enabled`: Enable/disable INDI control
- `scale`: INDI gain scaling factor (0.5-1.0)
- `blend_ratio`: 0=pure NDI, 1=pure INDI
- `alpha_omega_lp`: Low-pass filter coefficient for omega filtering
- `alpha_acc_lp`: Low-pass filter coefficient for angular acceleration

## Algorithm Flow

1. **Initialization Check**: If first run, initialize INDI state variables
2. **Torque Rotation**: Rotate previous torque using small-angle approximation
3. **Rate Filtering**: Apply two-stage filtering to angular rates
4. **Desired Acceleration**: Compute desired angular acceleration from rate errors
5. **INDI Correction**: Calculate torque increment with safeguards
6. **Yaw Rate Scaling**: Apply scaling based on current yaw rate
7. **Delta Tau Clamping**: Limit torque increments
8. **Blending**: Combine NDI and INDI commands
9. **Feedback Addition**: Add small feedback correction
10. **State Update**: Update variables for next iteration

## Safety Features

### Safeguards Implemented:
1. **NaN Detection**: Zero torques if NaN detected
2. **Increment Limiting**: ±3.0 N⋅m maximum increments
3. **Yaw Rate Scaling**: Automatic scaling at high yaw rates
4. **Blend Ratio**: Conservative mixing with proven NDI baseline
5. **Filter Bounds**: Bounded filter coefficients to prevent instability

## Performance Benefits

### Expected Improvements:
1. **Reduced Control Noise**: Filtered angular rate differentiation
2. **Better Disturbance Rejection**: INDI adapts to model uncertainties
3. **Stable High-Rate Performance**: Yaw-rate scaling prevents instability
4. **Physical Consistency**: Proper torque frame rotation
5. **Robust Operation**: Multiple safeguards and conservative tuning

## Monitoring and Tuning

### Key Logged Variables:
- `yaw_rate_scale`: Shows when high-yaw-rate scaling is active
- `omega_dot_meas_filt`: Filtered angular acceleration estimation
- `blend_ratio`: Current mixing ratio between NDI and INDI
- `tau_cmd`: Final torque commands

### Tuning Guidelines:
- **blend_ratio**: Start at 0.3-0.4, increase gradually if needed
- **alpha_omega_lp**: 0.1-0.2 range, higher = more filtering
- **alpha_acc_lp**: 0.2-0.3 range, higher = more filtering
- **scale**: 0.5-1.0 range, affects INDI authority

## Conclusion

This implementation provides a **robust hybrid NDI + INDI controller** that:
- Maintains NDI baseline performance
- Adds INDI benefits for disturbance rejection
- Includes multiple safety mechanisms
- Is fully configurable via JSON parameters
- Provides comprehensive logging for analysis

The changes are **minimal but impactful**, adding robust INDI capability while preserving the existing NDI foundation. The conservative default settings ensure safe operation while allowing tuning for performance optimization.