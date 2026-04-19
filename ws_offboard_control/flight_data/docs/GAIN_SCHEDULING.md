# Attitude Controller Gain Scheduling

## Overview
Implemented yaw rate-based gain scheduling for the attitude controller to improve stability during high-speed rotations. The gain scheduling dynamically adjusts the proportional (P) and derivative (D) gains based on the magnitude of the yaw rate.

## Motivation
During high yaw rate maneuvers (e.g., aggressive spins), the attitude controller benefits from:
- **Lower P gain**: Reduces aggressive corrections that can excite high-frequency modes
- **Higher D gain**: Provides damping to prevent oscillations and improves stability

This approach is particularly effective for:
- Spinning maneuvers with yaw rates > 5 rad/s (~286 deg/s)
- Reducing overshoot during rapid attitude changes
- Improving robustness to modeling uncertainties at high rates

## Implementation

### 1. Gain Scheduling Parameters
Added to class members:
```cpp
double Kp_att_base_{4.00};       // Baseline P gain at low yaw rates
double Kp_att_min_{1.50};        // Minimum P gain at high yaw rates
double Kd_att_base_{0.0};        // Baseline D gain at low yaw rates
double Kd_att_max_{0.60};        // Maximum D gain at high yaw rates
double gain_schedule_rate_thresh_{5.0};  // Yaw rate threshold (rad/s)
bool gain_scheduling_enabled_{true};
```

### 2. Scheduling Algorithm
In `controller_innerloop()`:
```cpp
double yaw_rate_mag = std::abs(r);  // r = body yaw rate (rad/s)
double schedule_factor = 0.0;  // 0 = low rate, 1 = high rate

if (gain_scheduling_enabled_ && yaw_rate_mag > 0.1) {
    // Smooth transition using tanh for gradual scheduling
    schedule_factor = std::tanh(yaw_rate_mag / gain_schedule_rate_thresh_);
    
    // Decrease P gain with increasing yaw rate
    Kp_att_ = Kp_att_base_ + (Kp_att_min_ - Kp_att_base_) * schedule_factor;
    
    // Increase D gain with increasing yaw rate
    Kd_att_ = Kd_att_base_ + (Kd_att_max_ - Kd_att_base_) * schedule_factor;
}
```

### 3. Scheduling Function
Uses `tanh()` for smooth, continuous transition:
- At yaw_rate = 0: `schedule_factor ≈ 0` → baseline gains
- At yaw_rate = 5 rad/s: `schedule_factor ≈ 0.76` → mostly scheduled gains
- At yaw_rate = 10 rad/s: `schedule_factor ≈ 0.96` → nearly full scheduled gains

### 4. Configuration
Added to `control_params.json`:
```json
"attitude": {
  "kp": 4.00,      // Baseline P gain (Kp_att_base)
  "kd": 0.0        // Baseline D gain (Kd_att_base)
},
"gain_scheduling": {
  "enabled": true,
  "kp_min": 1.50,           // Minimum P at high rates
  "kd_max": 0.60,           // Maximum D at high rates
  "rate_threshold": 5.0     // Threshold yaw rate (rad/s)
}
```

## Gain Behavior

### Low Yaw Rates (< 1 rad/s)
- **Kp_att** ≈ 4.00 (high P for quick response)
- **Kd_att** ≈ 0.00 (minimal damping)
- Behavior: Fast attitude corrections, minimal damping

### Medium Yaw Rates (2-5 rad/s)
- **Kp_att**: Smoothly decreases from 4.00 → 1.50
- **Kd_att**: Smoothly increases from 0.00 → 0.60
- Behavior: Gradual transition to more damped response

### High Yaw Rates (> 5 rad/s)
- **Kp_att** ≈ 1.50 (reduced P for stability)
- **Kd_att** ≈ 0.60 (high D for damping)
- Behavior: Stable, well-damped response during spins

## Expected Benefits

1. **Improved Stability**: Higher damping at high rates prevents oscillations
2. **Reduced Overshoot**: Lower P gain reduces aggressive corrections
3. **Better Tracking**: Smooth gain transition maintains good performance
4. **Robustness**: Adaptive gains handle varying operating conditions

## Monitoring

The gain scheduling status is logged every 80 control cycles:
```
GAIN_SCHED: yaw_rate=7.32 rad/s, schedule_factor=0.897, Kp_att=1.726, Kd_att=0.538
```

This shows:
- Current yaw rate magnitude
- Scheduling factor (0-1)
- Active Kp and Kd values

## Tuning Guidelines

### If oscillations occur at high yaw rates:
1. Increase `kd_max` (more damping)
2. Decrease `kp_min` (less aggressive corrections)

### If response is too slow at high yaw rates:
1. Increase `kp_min` (more aggressive corrections)
2. Decrease `kd_max` (less damping)

### If transition is too abrupt:
1. Increase `rate_threshold` (spread transition over wider range)

### To disable gain scheduling:
```json
"gain_scheduling": {
  "enabled": false
}
```

## Testing Recommendations

1. **Hover Test**: Verify baseline gains work well at low rates
2. **Slow Spin Test**: Check smooth transition (1-5 rad/s)
3. **Fast Spin Test**: Verify stability at high rates (>5 rad/s)
4. **Step Response**: Test attitude step commands at various yaw rates

## Integration with INDI

The gain scheduling works in parallel with INDI control:
- Gain scheduling affects the **attitude loop** (angle → rate commands)
- INDI affects the **rate loop** (rate commands → torques)
- Both contribute to overall stability and performance

## Future Enhancements

Potential improvements:
1. **Multi-axis scheduling**: Schedule gains based on total angular rate magnitude
2. **Adaptive thresholds**: Learn optimal rate thresholds during flight
3. **State-dependent scheduling**: Different gains for different flight phases
4. **Saturation-based scheduling**: Adjust gains when actuators saturate
