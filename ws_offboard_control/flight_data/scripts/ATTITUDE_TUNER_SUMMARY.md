# Summary: Attitude Auto-Tuner Created

## Files Created

### 1. `auto_tuner_attitude.py` (Main Tuner)
- **Purpose**: Specialized auto-tuner for attitude controller + gain scheduling
- **Optimizes**: 5 parameters (att_kp_base, att_kd_base, gs_kp_min, gs_kd_max, gs_rate_threshold)
- **Method**: Bayesian optimization with Gaussian Process
- **Cost function**: Heavily emphasizes attitude tracking (roll/pitch RMS errors)
- **Features**:
  - Smart caching (never re-evaluates same parameters)
  - History support (learns from previous runs)
  - Graceful interruption (Ctrl+C)
  - Automatic process cleanup
  - Separate evaluation of high-rate vs low-rate performance

### 2. `visualize_gain_scheduling.py` (Visualizer)
- **Purpose**: Visualize how gains change with yaw rate
- **Output**: 3 plots showing:
  - P gain vs yaw rate (decreases)
  - D gain vs yaw rate (increases)
  - Scheduling factor α vs yaw rate
- **Saves to**: `flight_data/outputs/gain_scheduling_plot.png`

### 3. Documentation
- `README_ATTITUDE_TUNER.md` - Full documentation
- `ATTITUDE_TUNER_QUICKSTART.md` - Quick reference

## Key Differences from Original `auto_tuner.py`

| Feature | Original `auto_tuner.py` | New `auto_tuner_attitude.py` |
|---------|-------------------------|------------------------------|
| **Parameters** | 6 (NDI, feedback, attitude) | 5 (attitude base + gain scheduling) |
| **Focus** | Full system tuning | Attitude control only |
| **Cost function** | Balanced (position, attitude, rate) | Attitude-heavy (30x weight on att_rms) |
| **Gain scheduling** | Not included | Primary focus |
| **Results file** | `tuning_results.json` | `tuning_results_attitude.json` |
| **High-rate analysis** | Not included | Separate high/low rate penalties |

## Gain Scheduling Implementation

The attitude controller C++ code was modified to implement gain scheduling:

```cpp
// Compute scheduling factor based on yaw rate
float alpha = std::min(std::abs(yaw_rate) / rate_threshold, 1.0f);

// Schedule gains: decrease P, increase D at high rates
float kp_scheduled = kp_base + alpha * (kp_min - kp_base);
float kd_scheduled = kd_base + alpha * (kd_max - kd_base);

// Use scheduled gains for control
torque_cmd = -kp_scheduled * att_error - kd_scheduled * rate_error;
```

## Configuration Updates

Added to `control_params.json`:
```json
{
  "gain_scheduling": {
    "enabled": true,
    "kp_min": 1.50,
    "kd_max": 0.60,
    "rate_threshold": 5.0
  }
}
```

## Usage Workflow

1. **Run tuner** (40 trials, ~2 hours):
   ```bash
   python3 auto_tuner_attitude.py --trials 40
   ```

2. **Visualize gains**:
   ```bash
   python3 visualize_gain_scheduling.py
   ```

3. **Test in flight**:
   ```bash
   ros2 run px4_ros_com offboard_control_spin_tt
   ```

4. **Analyze results**:
   ```bash
   python3 generate_plots.py
   ```

## Cost Function Breakdown

```python
cost = (
    30.0 * att_rms +              # Roll/pitch tracking (PRIMARY)
    5.0 * rate_rms +               # Roll/pitch rate tracking
    3.0 * yaw_rate_rms +           # Yaw rate tracking
    0.5 * torque_jerk +            # Control smoothness
    0.1 * control_effort +         # Efficiency
    10.0 * high_rate_att_err +     # Performance at high yaw rates
    5.0 * low_rate_att_err         # Performance at low yaw rates
)
```

**Key insight**: Separate penalties for high-rate (>3 rad/s) and low-rate (<1 rad/s) regions ensure good performance across the entire operating range.

## Expected Performance

Good parameter sets typically achieve:
- **Cost**: < 20 (excellent), 20-30 (good), >50 (needs work)
- **Attitude RMS**: < 0.03 rad
- **Rate RMS**: < 0.15 rad/s
- **Smooth torque commands**: Low jerk values

## Next Steps

1. **Initial run**: Start with 30-40 trials to explore parameter space
2. **Review results**: Check `tuning_results_attitude.json` for best parameters
3. **Visualize**: Run `visualize_gain_scheduling.py` to understand behavior
4. **Flight test**: Validate with actual spinning maneuvers
5. **Refine**: Run additional trials with `--use-history` to improve

## Tips

- **Start with history enabled** - learns from previous attempts
- **Monitor console output** - watch for "NEW BEST COST" messages
- **Check for crashes** - high costs (>100) indicate failures
- **Validate manually** - always test optimized parameters in real flight
- **Adjust search space** - if best hits boundaries, expand ranges in code

## Files Modified

- `src/px4_ros_com/src/examples/offboard/offboard_control_spin_tt.cpp`
  - Added gain scheduling implementation
  - Added parameters: `gs_kp_min`, `gs_kd_max`, `gs_rate_threshold`
  - Modified attitude controller to use scheduled gains
  - Added logging of scheduled gains vs base gains

## Build Required

After modifying the C++ code:
```bash
cd ~/ws_offboard_control
colcon build --packages-select px4_ros_com
source install/local_setup.bash
```

## Troubleshooting

**Issue**: No improvement over baseline
- **Solution**: Increase trials, check if baseline is already good

**Issue**: Costs too high (>100)
- **Solution**: Check for crashes, verify parameter ranges are reasonable

**Issue**: Gain scheduling not active
- **Solution**: Verify `"enabled": true` in `gain_scheduling` config section

**Issue**: Tuner gets stuck
- **Solution**: Try `--no-history` to start fresh exploration

## Performance Monitoring

After each trial, the tuner prints:
```
Attitude RMS: 0.0234 rad       <- Roll/pitch tracking
Rate RMS: 0.1456 rad/s          <- Rate tracking
Yaw rate RMS: 0.0823 rad/s      <- Yaw rate tracking
High-rate att err: 0.0289 rad   <- Performance at high rates
Low-rate att err: 0.0198 rad    <- Performance at low rates
Cost: 16.8923                   <- Total weighted cost
```

Monitor these values to understand what the optimizer is improving.

---

**Created**: October 16, 2025
**Author**: GitHub Copilot
**Purpose**: Optimize spinning maneuver attitude control with gain scheduling
