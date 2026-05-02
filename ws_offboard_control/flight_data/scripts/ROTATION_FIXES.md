# Rotation Transformation Fixes

## Summary of Changes

All rotation transformations in `visualize_flight.py` have been corrected to properly handle the coordinate frame conversions between PX4's NED frame and the visualization ENU frame.

## Key Issues Fixed

### 1. Rotation Matrix Construction (`rotation_matrix` function)
**Problem**: The function needed clearer documentation about the intrinsic ZYX (3-2-1) Euler angle convention used by PX4.

**Fix**: 
- Added comprehensive documentation explaining the ZYX intrinsic rotation order
- Clarified that the matrix transforms vectors from body frame to NED frame
- Documented the sign conventions: roll (right wing down positive), pitch (nose up positive), yaw (clockwise from above positive)
- Formula: `R_body_to_NED = R_z(yaw) × R_y(pitch) × R_x(roll)`

### 2. NED to ENU Coordinate Transformation (`transform_drone` function)
**Problem**: Used incorrect transformation `P @ R_body_to_ned @ P.T` which doesn't properly convert the rotation matrix.

**Fix**:
- Changed to correct chaining: `R_body_to_ENU = R_NED_to_ENU @ R_body_to_NED`
- Where `R_NED_to_ENU` is the rotation matrix that transforms NED vectors to ENU:
  ```
  [ENU_x]   [0  1  0] [NED_x]     [NED_y]   [East]
  [ENU_y] = [1  0  0] [NED_y]  =  [NED_x] = [North]
  [ENU_z]   [0  0 -1] [NED_z]     [-NED_z]  [Up]
  ```
- This properly composes the transformations: body → NED → ENU

### 3. Tilt Angle Computation (`_precompute_tilts_and_torques` method)
**Problem**: Tilt angles were computed using the wrong sign for the vertical component.

**Fix**:
- Body Z-axis transformed to ENU: `body_z_in_enu = R_body_to_ENU @ [0, 0, 1]`
- When drone is level, body Z points down, which in ENU is [0, 0, -1]
- Corrected formulas:
  - East tilt: `atan2(body_z_in_enu[0], -body_z_in_enu[2])`
  - North tilt: `atan2(body_z_in_enu[1], -body_z_in_enu[2])`
- The negative sign accounts for body Z pointing down when level

### 4. Torque Transformation
**Problem**: Torque vectors needed proper rotation from body frame to ENU world frame.

**Fix**:
- Apply the same rotation matrix: `tau_ENU = R_body_to_ENU @ tau_body`
- This correctly transforms both commanded and motor-computed torques
- Now torque arrows in visualization align with actual force directions in world frame

## Frame Convention Reference

### Body Frame (PX4/MAVLink convention)
- X-axis: Forward (North when aligned with NED)
- Y-axis: Right (East when aligned with NED)
- Z-axis: Down

### NED Frame (North-East-Down)
- X-axis: North
- Y-axis: East
- Z-axis: Down (towards Earth center)

### ENU Frame (East-North-Up) - Used for visualization
- X-axis: East
- Y-axis: North
- Z-axis: Up (away from Earth)

## Testing
Run the visualization with:
```bash
python3 visualize_flight.py ../logs/flight_log_YYYYMMDD_HHMMSS.csv [skip_frames]
```

Expected results:
- Drone body orientation should match actual flight attitude
- Tilt angles should correctly show North/East deviations
- Torque arrows should point in the correct world-frame directions
- Motor positions should be geometrically correct relative to drone center
