# Control Architecture Documentation

## Overview
This document describes the hierarchical control architecture for the quadrotor with spinning capability. The controller uses a cascaded structure with outer-loop position control, middle-loop attitude control, and inner-loop rate control with NDI/INDI methods.

---

## Control Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        OUTER LOOP (Position)                     │
│  Input: Position setpoint (x,y,z)_des                           │
│  Output: Desired acceleration a_des (body frame)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓ a_des
┌─────────────────────────────────────────────────────────────────┐
│                      MIDDLE LOOP (Attitude)                      │
│  Input: a_des → (roll, pitch)_des                               │
│  Output: Desired body rates ω_des = [p, q, r]_des               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓ ω_des
┌─────────────────────────────────────────────────────────────────┐
│                   INNER LOOP (Rate Control)                      │
│  Input: ω_des, ω_actual                                          │
│  Output: Torque commands τ = [τ_φ, τ_θ, τ_ψ]                   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ↓ τ
┌─────────────────────────────────────────────────────────────────┐
│                     ACTUATOR ALLOCATION                          │
│  Converts thrust + torque → Motor commands                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Outer Loop: Position Control

### Purpose
Converts position error into desired acceleration commands in the body frame.

### XY (Horizontal) Position Control

**State Variables:**
- Position: `(x, y)` (NED frame)
- Velocity: `(vx, vy)` (NED frame, low-pass filtered)
- Position error integral: `∫(x_des - x)dt`, `∫(y_des - y)dt`

**Control Law (for each axis, x or y):**

```
a_des_xy = Kp_pos_xy · (pos_des_xy - pos_xy) 
         - Kd_pos_xy · vel_xy_filt 
         + Ki_pos_xy · pos_err_int_xy
```

**Key Parameters:**
| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `Kp_pos_xy` | [0.55, 0.55] | Position proportional gain | ↑ for faster response, but may overshoot |
| `Kd_pos_xy` | [1.10, 1.10] | Velocity damping gain | ↑ to reduce oscillations |
| `Ki_pos_xy` | [0.03, 0.03] | Position integral gain | ↑ to eliminate steady-state error |
| `vel_lpf_alpha` | 0.15 | Velocity filter strength | ↓ for smoother velocity (more lag) |
| `max_horz_acc` | 3.0 m/s² | Maximum horizontal acceleration | Safety limit |
| `max_xy_acc_step` | 0.6 m/s² | Max acceleration change per cycle | Jerk limiting (smoothness) |

**Saturation & Anti-Windup:**
- Acceleration is clamped to ±`max_horz_acc`
- If saturated, the integral term stops accumulating (anti-windup)

---

### Z (Vertical) Position Control

**Control Law:**

```
a_des_z = Kp_pos_z · (z_des - z) 
        - Kd_pos_z · vz 
        + Ki_pos_z · ∫(z_des - z)dt 
        - g
```

**Key Parameters:**
| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `Kp_pos_z` | 20.0 | Z position gain | ↑ for aggressive altitude hold |
| `Kd_pos_z` | 20.0 | Z velocity damping | ↑ to reduce altitude oscillations |
| `Ki_pos_z` | 1.0 | Z integral gain | Small value to compensate hover thrust |
| `max_vert_acc` | 9.81 m/s² | Max vertical acceleration | Usually set to gravity |

**Note:** The term `-g` compensates for gravity in NED frame (down is positive).

---

### Acceleration to Thrust Conversion

The desired acceleration `a_des` (body frame) is converted to normalized thrust:

```
thrust_norm = a_des / (2 * g)
```

Where:
- Each component is clamped to [-1.0, 1.0]
- This is then published as `VehicleThrustSetpoint`

---

## 2. Middle Loop: Attitude Control

### Purpose
Converts desired acceleration into desired attitude (roll, pitch) and then into desired body rates.

### Desired Attitude Calculation

From the desired acceleration vector `a_des = [ax, ay, az]`:

```
pitch_des = -atan2(ax, √(az² + ay²))
roll_des  = atan2(ay, |az|)
```

**Saturation:**
Both angles are clamped to ±`max_tilt_angle` (default 0.7 rad ≈ 40°)

---

### Euler Rate Controller

The attitude error is controlled via desired Euler angle rates:

```
φ̇_des = Kp_att · (roll_des  - roll)
θ̇_des = Kp_att · (pitch_des - pitch)
ψ̇_des = 0  (yaw uncontrolled for spin mode)
```

**Key Parameters:**
| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `Kp_att_base` | 4.00 | Baseline attitude P gain | ↑ for faster attitude response |
| `Kd_att_base` | 0.0 | Baseline rate damping | ↑ to add damping (smooth oscillations) |
| `max_tilt_angle` | 0.7 rad | Max roll/pitch angle | Safety limit |

---

### Gain Scheduling (Advanced Feature)

At high yaw rates (spinning), the controller adjusts gains to maintain stability:

```
schedule_factor = (tanh((|r| - rate_thresh) / 3 * 4) + 1) / 2

Kp_att = Kp_att_base + (Kp_att_min - Kp_att_base) · schedule_factor
Kd_att = Kd_att_base + (Kd_att_max - Kd_att_base) · schedule_factor
```

**Key Parameters:**
| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `gain_scheduling_enabled` | true | Enable/disable scheduling | Turn off for debugging |
| `Kp_att_min` | 1.50 | Min P gain at high yaw rate | Lower for stability during spin |
| `Kd_att_max` | 0.60 | Max D gain at high yaw rate | Higher damping during spin |
| `gain_schedule_rate_thresh` | 5.0 rad/s | Yaw rate threshold | Rate where scheduling activates |

---

### Euler Rates → Body Rates

Convert Euler angle rates to body frame angular rates using the transformation:

```
T_inv = [ 1,   0,      -sin(θ)  ]
        [ 0,  cos(φ),  sin(φ)cos(θ) ]
        [ 0, -sin(φ),  cos(φ)cos(θ) ]

ω_des = T_inv · [φ̇_des, θ̇_des, ψ̇_des]ᵀ - Kd_att · ω_actual
```

Where `ω_des = [p_des, q_des, r_des]ᵀ` (desired roll/pitch/yaw rates)

**The second term** `-Kd_att · ω_actual` **adds rate feedback damping.**

---

## 3. Inner Loop: Rate Control (NDI/INDI)

### Purpose
Track desired body rates by commanding torques using Nonlinear Dynamic Inversion (NDI) with optional Incremental NDI (INDI).

---

### Method 1: Standard NDI (Default)

**Rotational Dynamics:**

```
İ · ω̇ = τ + ω × (I · ω)
```

Where:
- `I` = Inertia matrix (diagonal: [Ix, Iy, Iz])
- `ω` = [p, q, r] (body rates)
- `τ` = [τ_φ, τ_θ, τ_ψ] (torque commands)

**Control Law:**

First, calculate desired angular acceleration:

```
ω̇_des = Kp_ndi · (ω_des - ω_filt) + Kd_ndi · d/dt(ω_des - ω_filt)
```

Then, invert the dynamics to find required torque:

```
τ_NDI = I · ω̇_des + ω × (I · ω)
```

**Feedback Correction (robustness):**

```
τ_fb = Kp_fb · (ω_des - ω) + Kd_fb · d/dt(ω_des - ω)

τ_total = τ_NDI + τ_fb
```

**Key Parameters:**
| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `Kp_ndi_rate` | 0.100 | NDI rate P gain | ↑ for faster rate tracking |
| `Kd_ndi_rate` | 0.000 | NDI rate D gain | Usually small (derivatives are noisy) |
| `Kp_fb_rate` | 0.180 | Feedback correction P | Compensates for model errors |
| `Kd_fb_rate` | 0.010 | Feedback correction D | Adds damping to rate loop |
| `omega_lpf_alpha` | 0.1 | Rate filter for differentiation | ↓ for smoother derivatives |

---

### Method 2: Incremental NDI (INDI)

**Concept:** Uses the previous torque command and measured angular acceleration to improve tracking without perfect model knowledge.

**Control Law:**

```
ω̇_meas_filt = LPF(dω/dt)  # Measured angular acceleration

Δτ = I · (ω̇_des - ω̇_meas_filt)  # Incremental correction

τ_INDI = τ_prev + K_indi_scale · Δτ

τ_total = (1 - blend_ratio) · τ_NDI + blend_ratio · τ_INDI + τ_fb
```

**Key Parameters:**
| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `indi_enabled` | false | Enable INDI mode | True for high-rate spinning |
| `K_indi_scale` | 0.75 | INDI correction scale | 0.5-1.0 (higher = more aggressive) |
| `indi_blend_ratio` | 0.3 | NDI/INDI blend (0=NDI, 1=INDI) | Start with 0.3, increase gradually |
| `alpha_omega_lp` | 0.15 | Rate filter before differentiation | ↓ for smoother accel estimate |
| `alpha_acc_lp` | 0.25 | Angular acceleration filter | ↓ to reduce noise |

**When to use INDI:**
- High yaw rates (>10 rad/s)
- Model uncertainty (unknown inertia)
- Improved disturbance rejection

---

### Spin Mode

When `spin_mode_ = true`, an additional yaw torque is added:

```
τ_ψ_total = τ_ψ + spin_torque_cmd
```

**Key Parameter:**
| Parameter | Default | Description | Tuning Tips |
|-----------|---------|-------------|-------------|
| `spin_torque_cmd` | 2.0 N·m | Additional yaw torque for spinning | ↑ for faster spin, ↓ if unstable |

---

## Cumulative Control Equations

### Complete Thrust Command

```
F_body = m · (a_des + [0, 0, g])  # Total force in body frame

thrust_norm = a_des / (2g)  # Normalized thrust for PX4 [-1, 1]
```

Where `a_des` comes from outer loop:

```
a_des_world = Kp_pos · e_pos - Kd_pos · v_filt + Ki_pos · ∫e_pos dt

a_des_body = R_bw · a_des_world  # Rotate to body frame
```

---

### Complete Torque Command

**NDI Mode:**

```
ω_filt = LPF(ω_actual)
e_ω = ω_des - ω_filt
ė_ω = d/dt(e_ω)

ω̇_des = Kp_ndi · e_ω + Kd_ndi · ė_ω

τ_NDI = I · ω̇_des + ω × (I · ω)

τ_fb = Kp_fb · (ω_des - ω) + Kd_fb · d/dt(ω_des - ω)

τ_total = τ_NDI + τ_fb + [0, 0, τ_spin]  # If spin mode enabled
```

**INDI Mode:**

```
ω̇_meas = LPF(dω/dt)

Δτ = I · (ω̇_des - ω̇_meas)

τ_INDI = τ_prev + K_indi · Δτ

τ_total = (1 - α) · τ_NDI + α · τ_INDI + τ_fb + [0, 0, τ_spin]
```

Where `α = indi_blend_ratio`

---

## Tuning Guide

### Step-by-Step Tuning Process

#### 1. Start with Altitude Hold (Z-axis)
- Disable XY control (set `Kp_pos_xy = [0, 0]`)
- Tune `Kp_pos_z` first (increase until stable hover with small oscillations)
- Add `Kd_pos_z` to dampen oscillations
- Fine-tune `Ki_pos_z` for steady-state accuracy

#### 2. Tune Attitude Loop (Roll/Pitch)
- Set `Kp_att_base` to 3.0-5.0 (start conservative)
- Disable gain scheduling initially (`gain_scheduling_enabled = false`)
- Perform small angle commands and observe response
- If oscillatory, reduce `Kp_att` or increase `Kd_att`

#### 3. Tune Rate Controller (Inner Loop)
- Start with pure NDI (`indi_enabled = false`)
- Set `Kp_ndi_rate` to 0.05-0.15
- Set `Kd_ndi_rate` to 0 initially
- Add `Kp_fb_rate` (0.1-0.2) for model error compensation
- If unstable at high rates, reduce `Kp_ndi_rate`

#### 4. Enable Position Control (XY)
- Start with low `Kp_pos_xy` (0.3-0.5)
- Increase `Kd_pos_xy` to match (0.6-1.2)
- Add small `Ki_pos_xy` (0.01-0.05) for tracking

#### 5. Test Spin Mode
- Enable spin with small `spin_torque_cmd` (0.5-1.0 N·m)
- Enable gain scheduling if unstable (`gain_scheduling_enabled = true`)
- Adjust `Kp_att_min` and `Kd_att_max` for spin stability
- Consider enabling INDI for very high spin rates

#### 6. Optimize INDI (Optional)
- Enable INDI (`indi_enabled = true`)
- Start with `indi_blend_ratio = 0.2` (mostly NDI)
- Gradually increase to 0.3-0.5
- Tune `K_indi_scale` (0.5-1.0)

---

## Parameter Sensitivity Matrix

| Symptom | Likely Cause | Parameter to Adjust |
|---------|--------------|---------------------|
| Altitude oscillations | Z-axis overdamped | ↓ `Kd_pos_z` |
| Slow altitude response | Z-axis underdamped | ↑ `Kp_pos_z` |
| Altitude drift | Poor integral action | ↑ `Ki_pos_z` |
| Horizontal oscillations | XY overdamped or high gains | ↓ `Kp_pos_xy` or ↑ `Kd_pos_xy` |
| Slow horizontal response | XY underdamped | ↑ `Kp_pos_xy` |
| Attitude wobble | Attitude loop unstable | ↓ `Kp_att` or ↑ `Kd_att` |
| Slow attitude correction | Attitude loop too slow | ↑ `Kp_att` |
| Rate tracking error | NDI gains too low | ↑ `Kp_ndi_rate` or `Kp_fb_rate` |
| High-frequency oscillations | Rate loop noisy | ↑ `omega_lpf_alpha` (more filtering) |
| Unstable during spin | Need gain scheduling | Enable scheduling or ↓ `Kp_att_min` |
| Model mismatch errors | NDI model inaccurate | Enable INDI or ↑ `Kp_fb_rate` |

---

## Configuration File Example

Create `/home/pyro/ws_offboard_control/flight_data/config/control_params.json`:

```json
{
  "position_xy": {
    "kp_x": 0.55,
    "kp_y": 0.55,
    "kd_x": 1.10,
    "kd_y": 1.10,
    "ki_x": 0.03,
    "ki_y": 0.03
  },
  "position_z": {
    "kp": 20.0,
    "kd": 20.0,
    "ki": 1.0
  },
  "attitude": {
    "kp": 4.00,
    "kd": 0.0
  },
  "gain_scheduling": {
    "enabled": true,
    "kp_min": 1.50,
    "kd_max": 0.60,
    "rate_threshold": 5.0
  },
  "ndi_rate": {
    "kp": 0.100,
    "kd": 0.000
  },
  "feedback_rate": {
    "kp": 0.180,
    "kd": 0.010
  },
  "indi": {
    "enabled": false,
    "scale": 0.75,
    "blend_ratio": 0.3,
    "alpha_omega_lp": 0.15,
    "alpha_acc_lp": 0.25
  },
  "limits": {
    "max_horz_acc": 3.0,
    "max_tilt_angle": 0.7
  }
}
```

---

## Debugging Tips

1. **Check log files:** CSV files contain all errors, commands, and saturation flags
2. **Plot rate tracking:** Compare `p_des, q_des, r_des` vs `p, q, r`
3. **Monitor saturation:** `rate_sat_*` and `acc_sat_*` flags indicate limits hit
4. **Watch motor outputs:** If motors saturate (≈1.0 or 0.0), reduce gains or limits
5. **Enable verbose logging:** Set `offboard_setpoint_counter_ % 10 == 0` for more frequent prints

---

## Key Takeaways

1. **Cascaded structure:** Position → Attitude → Rate ensures stability at each level
2. **NDI inverts dynamics:** Requires accurate inertia matrix
3. **INDI adds robustness:** Uses feedback from accelerometers to handle model errors
4. **Gain scheduling:** Adapts controller to high yaw rates automatically
5. **Tuning priority:** Start from inner loop (rate) → middle (attitude) → outer (position)

---

## References

- **NDI Control Theory:** Stevens & Lewis, "Aircraft Control and Simulation"
- **INDI Method:** Grondman et al., "Incremental Nonlinear Dynamic Inversion Control"
- **PX4 Documentation:** https://docs.px4.io/main/en/flying/offboard_control.html

---

**Document Version:** 1.0  
**Last Updated:** October 17, 2025  
**Author:** Generated for `offboard_control_spin_tt.cpp`
