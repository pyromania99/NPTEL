# Attitude Controller Auto-Tuner

## Overview

`auto_tuner_attitude.py` is a specialized auto-tuner that focuses on optimizing **gain scheduling parameters only**:
1. **gs_kp_min** - Minimum P gain at high yaw rates
2. **gs_kd_max** - Maximum D gain at high yaw rates

The base attitude gains (kp_base, kd_base) and rate threshold are **kept fixed** from the config file. This makes tuning faster and more focused on finding optimal gain scheduling behavior.

## Gain Scheduling Strategy

The gain scheduling reduces the P term and increases the D term as yaw rate increases:

- **Low yaw rates (hovering)**: Use base gains (kp_base, kd_base)
- **High yaw rates (spinning)**: Use scheduled gains (kp_min, kd_max)
- **In between**: Linear interpolation based on rate threshold

### Scheduling Formula

```
alpha = clip(|yaw_rate| / rate_threshold, 0, 1)
kp_scheduled = kp_base + alpha * (kp_min - kp_base)
kd_scheduled = kd_base + alpha * (kd_max - kd_base)
```

### Rationale

- **Decrease P gain at high rates**: Reduces overshoot and oscillations during aggressive maneuvers
- **Increase D gain at high rates**: Provides damping to stabilize high-speed rotations
- **Smooth transition**: Linear interpolation prevents sudden gain changes

## Parameters Being Optimized

| Parameter | Range | Description | Status |
|-----------|-------|-------------|--------|
| `gs_kp_min` | 0.3 - 2.0 | Min P gain (at high rates) | **OPTIMIZED** |
| `gs_kd_max` | 0.3 - 1.5 | Max D gain (at high rates) | **OPTIMIZED** |

## Parameters Kept Fixed

| Parameter | Description | Source |
|-----------|-------------|--------|
| `att_kp_base` | Base P gain (at low rates) | From config file |
| `att_kd_base` | Base D gain (at low rates) | From config file |
| `gs_rate_threshold` | Rate threshold for full scheduling (rad/s) | From config file |

**Why only 2 parameters?**
- Faster optimization (30 trials instead of 40+)
- More focused search space
- Base gains should already be tuned for hover/low-rate performance
- Focuses specifically on high-rate spinning behavior

## Cost Function

The cost function emphasizes attitude tracking performance:

```python
cost = (
    30.0 * att_rms +              # PRIMARY: Roll/pitch tracking
    5.0 * rate_rms +               # Rate tracking (roll/pitch)
    3.0 * yaw_rate_rms +           # Yaw rate tracking
    0.5 * torque_jerk +            # Smoothness
    0.1 * control_effort +         # Efficiency
    10.0 * high_rate_att_err +     # High-rate performance
    5.0 * low_rate_att_err         # Low-rate performance
)
```

This weights:
- Attitude tracking errors heavily (primary objective)
- Separate penalties for high-rate and low-rate performance
- Smooth control (penalize oscillations)
- Control efficiency

## Usage

### Basic Usage

```bash
cd /home/pyro/ws_offboard_control/flight_data/scripts
python3 auto_tuner_attitude.py --trials 30
```

**Note**: Only 30 trials needed now (was 40) since we're only optimizing 2 parameters!

### Options

```bash
# Use history to warm start (default)
python3 auto_tuner_attitude.py --use-history --trials 30

# Start fresh (ignore previous results)
python3 auto_tuner_attitude.py --no-history --trials 30

# Control which historical results seed the optimizer
python3 auto_tuner_attitude.py --max-seed-cost 50 --trials 30

# Run more trials for thorough search
python3 auto_tuner_attitude.py --trials 40

# Quick test run
python3 auto_tuner_attitude.py --trials 15
```

### Graceful Interruption

Press Ctrl+C once to stop after the current trial completes.
Press Ctrl+C twice to force immediate exit.

## Visualizing Gain Scheduling

To visualize how gains change with yaw rate:

```bash
cd /home/pyro/ws_offboard_control/flight_data/scripts
python3 visualize_gain_scheduling.py
```

This creates plots showing:
1. P gain vs yaw rate
2. D gain vs yaw rate
3. Scheduling factor α vs yaw rate

Output: `/home/pyro/ws_offboard_control/flight_data/outputs/gain_scheduling_plot.png`

## Results

Results are saved to:
```
/home/pyro/ws_offboard_control/flight_data/outputs/tuning_results_attitude.json
```

The file contains:
- `best_cost`: Best cost achieved
- `best_params`: Best parameter values found
- `trial_history`: Complete history of all trials (deduplicated)

## Workflow

**Prerequisites**: Make sure your base attitude gains are already tuned for good hover/low-rate performance!

1. **Initial Tuning**: Run with fresh start to explore parameter space
   ```bash
   python3 auto_tuner_attitude.py --no-history --trials 25
   ```

2. **Refinement**: Run with history to refine around good solutions
   ```bash
   python3 auto_tuner_attitude.py --use-history --trials 15
   ```

3. **Visualization**: Check gain scheduling behavior
   ```bash
   python3 visualize_gain_scheduling.py
   ```

4. **Flight Testing**: Test with actual spinning maneuvers
   ```bash
   ros2 run px4_ros_com offboard_control_spin_tt
   ```

5. **Analysis**: Check attitude tracking performance
   ```bash
   python3 generate_plots.py
   ```

## Configuration File

The tuner modifies:
```
/home/pyro/ws_offboard_control/flight_data/config/control_params.json
```

Specifically:
- `attitude.kp` - Base P gain
- `attitude.kd` - Base D gain
- `gain_scheduling.enabled` - Enable/disable gain scheduling
- `gain_scheduling.kp_min` - Min P at high rates
- `gain_scheduling.kd_max` - Max D at high rates
- `gain_scheduling.rate_threshold` - Rate threshold

## Tips

1. **Start with history**: The tuner learns from previous trials
2. **Monitor costs**: Good costs are typically < 20 for spinning maneuvers
3. **Check visualizations**: Use `visualize_gain_scheduling.py` to understand behavior
4. **Validate manually**: Always test optimized parameters with real flights
5. **Adjust ranges**: If best values hit boundaries, expand search space in the code

## Troubleshooting

### No log files generated
- Check that PX4 SITL and Gazebo are properly installed
- Verify controller builds successfully: `colcon build --packages-select px4_ros_com`

### High costs (>100)
- Check that the drone is actually flying (not crashing)
- Verify control_params.json has reasonable values
- Check for process conflicts (kill stray PX4/Gazebo instances)

### Optimization seems stuck
- Try `--no-history` to start fresh
- Reduce `--max-seed-cost` to only use best historical points
- Check that parameter ranges make sense for your drone

## Related Scripts

- `auto_tuner.py` - Full system tuner (NDI, feedback, attitude, position)
- `yaw_ff_tuner.py` - Yaw feedforward predictor tuner
- `generate_plots.py` - Generate analysis plots from flight logs
- `visualize_flight.py` - 3D flight visualization

## Example Output

```
======================================================================
ATTITUDE CONTROLLER AUTO-TUNER
======================================================================
Optimizing: attitude base gains + gain scheduling
Trials: 40
======================================================================

Loaded 15 seed points from 32 trials. Cache: 28
Warm-starting from historical best (cost=18.2456)

Trial 1
======================================================================
Parameters:
  att_kp_base         : 1.3791
  att_kd_base         : 3.6541
  gs_kp_min           : 0.8500
  gs_kd_max           : 0.5200
  gs_rate_threshold   : 5.0000

...flight test runs...

  Attitude RMS: 0.0234 rad
  Rate RMS: 0.1456 rad/s
  Yaw rate RMS: 0.0823 rad/s
  High-rate att err: 0.0289 rad
  Low-rate att err: 0.0198 rad

*** NEW BEST COST: 16.8923 ***

Cost: 16.8923
```

## Notes

- The tuner automatically restarts PX4/Gazebo between trials
- Cache prevents re-evaluation of identical parameters
- Results are merged with historical data on save
- Best parameters are automatically applied to the config file
