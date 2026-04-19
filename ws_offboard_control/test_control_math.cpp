/**
 * @brief Test script for control math calculations
 * @file test_control_math.cpp
 * 
 * Tests the horizontal control disable and attitude calculations from body accelerations
 * Uses the same Eigen library and math functions as offboard_control_spin_tt.cpp
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Helper function to limit a value
template<typename T>
void limit(T &val, T min_val, T max_val) {
    if (val < min_val) val = min_val;
    if (val > max_val) val = max_val;
}

// Convert quaternion to Euler angles (same as in your code)
void quaternion_to_euler(double qw, double qx, double qy, double qz,
                        float &roll, float &pitch, float &yaw) {
    // Roll (x-axis rotation)
    double sinr_cosp = 2.0 * (qw * qx + qy * qz);
    double cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (y-axis rotation)
    double sinp = 2.0 * (qw * qy - qz * qx);
    if (std::abs(sinp) >= 1)
        pitch = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        pitch = std::asin(sinp);

    // Yaw (z-axis rotation)
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    yaw = std::atan2(siny_cosp, cosy_cosp);
}

// Convert Euler angles to quaternion
Eigen::Quaterniond euler_to_quaternion(double roll, double pitch, double yaw) {
    Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
    
    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    return q;
}

void print_separator(const std::string& title = "") {
    std::cout << std::string(80, '=') << std::endl;
    if (!title.empty()) {
        std::cout << title << std::endl;
        std::cout << std::string(80, '=') << std::endl;
    }
}

void test_horizontal_control_disable() {
    print_separator("TEST 1: Horizontal Control Disable");
    
    // Setup test scenario
    Eigen::Vector2d pos_err_xy(2.0, -1.5);  // Some position error
    Eigen::Vector2d vel_xy_filt(0.5, -0.3);  // Some velocity
    
    // Controller gains (from your code)
    double K_p_xy = 1.5;
    double K_v_xy = 2.0;
    
    // With horizontal control enabled (normal operation)
    Eigen::Vector2d a_des_world_xy_enabled = -K_p_xy * pos_err_xy - K_v_xy * vel_xy_filt;
    
    // With horizontal control disabled (TEMPORARY mode)
    Eigen::Vector2d a_des_world_xy_disabled(0.0, 0.0);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nPosition Error XY:    [" << pos_err_xy(0) << ", " << pos_err_xy(1) << "] m" << std::endl;
    std::cout << "Velocity XY:          [" << vel_xy_filt(0) << ", " << vel_xy_filt(1) << "] m/s" << std::endl;
    std::cout << "\nAccel (enabled):      [" << a_des_world_xy_enabled(0) << ", " 
              << a_des_world_xy_enabled(1) << "] m/s²" << std::endl;
    std::cout << "Accel (disabled):     [" << a_des_world_xy_disabled(0) << ", " 
              << a_des_world_xy_disabled(1) << "] m/s²" << std::endl;
    std::cout << "\n✓ Horizontal control successfully disabled: a_xy = [0, 0]" << std::endl;
}

void test_body_frame_transformation() {
    print_separator("\nTEST 2: World to Body Frame Transformation");
    
    struct TestCase {
        std::string name;
        double roll, pitch, yaw;
    };
    
    std::vector<TestCase> test_cases = {
        {"Hover (no tilt)", 0.0, 0.0, 0.0},
        {"Roll 10°", M_PI/18.0, 0.0, 0.0},
        {"Pitch 15°", 0.0, M_PI/12.0, 0.0},
        {"Yaw 45°", 0.0, 0.0, M_PI/4.0},
        {"Combined", M_PI/18.0, M_PI/12.0, M_PI/6.0}
    };
    
    // Horizontal control disabled
    Eigen::Vector2d a_des_world_xy(0.0, 0.0);
    double a_des_z = -9.81;  // Gravity compensation
    
    std::cout << std::fixed << std::setprecision(2);
    
    for (const auto& tc : test_cases) {
        std::cout << "\n" << tc.name << ":" << std::endl;
        std::cout << "  Attitude: roll=" << tc.roll*180/M_PI << "°, pitch=" 
                  << tc.pitch*180/M_PI << "°, yaw=" << tc.yaw*180/M_PI << "°" << std::endl;
        
        // Create quaternion and rotation matrices (SAME AS YOUR CODE)
        Eigen::Quaterniond q = euler_to_quaternion(tc.roll, tc.pitch, tc.yaw);
        Eigen::Matrix3d R_wb = q.toRotationMatrix();   // body -> world
        Eigen::Matrix3d R_bw = R_wb.transpose();       // world -> body
        
        // Combine XY and Z into world acceleration vector (SAME AS YOUR CODE)
        Eigen::Vector3d a_des_world(a_des_world_xy(0), a_des_world_xy(1), a_des_z);
        
        // Transform to body frame (SAME AS YOUR CODE)
        Eigen::Vector3d a_des_body = R_bw * a_des_world;
        
        std::cout << "  World accel:  [" << a_des_world(0) << ", " << a_des_world(1) 
                  << ", " << a_des_world(2) << "] m/s²" << std::endl;
        std::cout << "  Body accel:   [" << a_des_body(0) << ", " << a_des_body(1) 
                  << ", " << a_des_body(2) << "] m/s²" << std::endl;
    }
}

void test_jerk_limit() {
    print_separator("\nTEST 3: Jerk/Slew Limit on XY Accelerations");
    
    // Simulate body accelerations
    Eigen::Vector3d a_des_body(5.0, -3.5, -9.81);  // Sudden large command
    Eigen::Vector2d a_des_prev_xy(0.0, 0.0);  // Starting from zero
    
    double max_xy_acc_step = 2.0;  // m/s² per timestep (from your code)
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nMax accel step:       " << max_xy_acc_step << " m/s²" << std::endl;
    std::cout << "Previous accel XY:    [" << a_des_prev_xy(0) << ", " 
              << a_des_prev_xy(1) << "] m/s²" << std::endl;
    std::cout << "Desired accel body:   [" << a_des_body(0) << ", " << a_des_body(1) 
              << ", " << a_des_body(2) << "] m/s²" << std::endl;
    
    // Jerk / slew limit on XY accelerations (EXACT CODE FROM YOUR IMPLEMENTATION)
    Eigen::Vector2d a_des_body_xy(a_des_body(0), a_des_body(1));
    for (int i = 0; i < 2; ++i) {
        double delta = a_des_body_xy(i) - a_des_prev_xy(i);
        limit(delta, -max_xy_acc_step, max_xy_acc_step);
        a_des_body_xy(i) = a_des_prev_xy(i) + delta;
        
        std::cout << "  Axis " << i << ": delta_orig=" << (a_des_body(i) - a_des_prev_xy(i))
                  << ", delta_limited=" << delta << std::endl;
    }
    
    std::cout << "\nLimited accel XY:     [" << a_des_body_xy(0) << ", " 
              << a_des_body_xy(1) << "] m/s²" << std::endl;
    std::cout << "✓ Jerk limiting prevents sudden changes" << std::endl;
}

void test_attitude_calculation() {
    print_separator("\nTEST 4: Attitude Calculation from Body Accelerations");
    
    struct TestCase {
        std::string name;
        double a_x, a_y, a_z;
    };
    
    std::vector<TestCase> test_cases = {
        {"Hover", 0.0, 0.0, -9.81},
        {"Forward", 2.0, 0.0, -9.81},
        {"Right", 0.0, 2.0, -9.81},
        {"Forward-Right", 2.0, 2.0, -9.81},
        {"Backward-Left", -2.0, -2.0, -9.81},
        {"Horizontal Disabled", 0.0, 0.0, -9.81}
    };
    
    std::cout << std::fixed << std::setprecision(2);
    
    for (const auto& tc : test_cases) {
        std::cout << "\n" << tc.name << ":" << std::endl;
        std::cout << "  Body accel: [" << tc.a_x << ", " << tc.a_y << ", " << tc.a_z << "] m/s²" << std::endl;
        
        // Calculate desired pitch and roll (EXACT FORMULAS FROM YOUR CODE)
        double pitch_des = -atan2(tc.a_x, sqrt(tc.a_z*tc.a_z + tc.a_y*tc.a_y));
        double roll_des = atan2(tc.a_y, std::abs(tc.a_z));
        
        std::cout << "  Pitch desired: " << pitch_des*180/M_PI << "°" << std::endl;
        std::cout << "  Roll desired:  " << roll_des*180/M_PI << "°" << std::endl;
        
        // Small angle approximation check
        if (std::abs(tc.a_x) < 5 && std::abs(tc.a_y) < 5) {
            double pitch_approx = tc.a_x / 9.81;
            double roll_approx = tc.a_y / 9.81;
            std::cout << "  Small angle approx: pitch≈" << pitch_approx*180/M_PI 
                      << "°, roll≈" << roll_approx*180/M_PI << "°" << std::endl;
        }
    }
}

void test_complete_pipeline() {
    print_separator("\nTEST 5: Complete Control Pipeline");
    
    // Simulation parameters
    double dt = 0.01;  // 100 Hz
    int n_steps = 500;  // 5 seconds
    
    // Controller parameters (from your code)
    double K_p_xy = 1.5;
    double K_v_xy = 2.0;
    double max_xy_acc_step = 2.0;
    
    // Storage
    std::vector<double> time_vec;
    std::vector<double> roll_des_vec;
    std::vector<double> pitch_des_vec;
    std::vector<double> a_x_vec;
    std::vector<double> a_y_vec;
    
    // Initial conditions
    Eigen::Vector2d a_des_prev_xy(0.0, 0.0);
    double current_roll = 0.0;
    double current_pitch = 0.0;
    double current_yaw = 0.0;
    
    // Reference trajectory: simulate tracking with some position error
    double radius = 2.0;
    double omega = 0.5;  // rad/s
    
    std::cout << "\nRunning " << n_steps << " timesteps at " << 1.0/dt << " Hz..." << std::endl;
    
    for (int i = 0; i < n_steps; ++i) {
        double t = i * dt;
        
        // Simulated position error (circular trajectory)
        Eigen::Vector2d pos_err_xy(radius * cos(omega * t), radius * sin(omega * t));
        Eigen::Vector2d vel_xy_filt(-radius * omega * sin(omega * t), 
                                    radius * omega * cos(omega * t));
        
        // OPTION 1: Horizontal control enabled (normal)
        // Eigen::Vector2d a_des_world_xy = -K_p_xy * pos_err_xy - K_v_xy * vel_xy_filt;
        
        // OPTION 2: Horizontal control disabled (TESTING)
        Eigen::Vector2d a_des_world_xy(0.0, 0.0);
        
        // World to body transformation (SAME AS YOUR CODE)
        Eigen::Quaterniond q = euler_to_quaternion(current_roll, current_pitch, current_yaw);
        Eigen::Matrix3d R_wb = q.toRotationMatrix();   // body -> world
        Eigen::Matrix3d R_bw = R_wb.transpose();       // world -> body
        
        double a_des_z = -9.81;
        Eigen::Vector3d a_des_world(a_des_world_xy(0), a_des_world_xy(1), a_des_z);
        Eigen::Vector3d a_des_body = R_bw * a_des_world;
        
        // Jerk / slew limit (SAME AS YOUR CODE)
        Eigen::Vector2d a_des_body_xy(a_des_body(0), a_des_body(1));
        for (int j = 0; j < 2; ++j) {
            double delta = a_des_body_xy(j) - a_des_prev_xy(j);
            limit(delta, -max_xy_acc_step, max_xy_acc_step);
            a_des_body_xy(j) = a_des_prev_xy(j) + delta;
        }
        a_des_prev_xy = a_des_body_xy;
        
        // Attitude calculation (SAME AS YOUR CODE)
        double pitch_des = -atan2(a_des_body_xy(0), sqrt(a_des_z*a_des_z + a_des_body_xy(1)*a_des_body_xy(1)));
        double roll_des = atan2(a_des_body_xy(1), std::abs(a_des_z));
        
        // Store data
        time_vec.push_back(t);
        roll_des_vec.push_back(roll_des * 180 / M_PI);
        pitch_des_vec.push_back(pitch_des * 180 / M_PI);
        a_x_vec.push_back(a_des_body_xy(0));
        a_y_vec.push_back(a_des_body_xy(1));
        
        // Simple attitude dynamics (for simulation)
        current_roll = 0.9 * current_roll + 0.1 * roll_des;
        current_pitch = 0.9 * current_pitch + 0.1 * pitch_des;
    }
    
    // Calculate statistics
    double max_a_x = 0.0, max_a_y = 0.0, max_roll = 0.0, max_pitch = 0.0;
    for (size_t i = 0; i < time_vec.size(); ++i) {
        max_a_x = std::max(max_a_x, std::abs(a_x_vec[i]));
        max_a_y = std::max(max_a_y, std::abs(a_y_vec[i]));
        max_roll = std::max(max_roll, std::abs(roll_des_vec[i]));
        max_pitch = std::max(max_pitch, std::abs(pitch_des_vec[i]));
    }
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nSummary Statistics:" << std::endl;
    std::cout << "  Max |a_x|:     " << max_a_x << " m/s²" << std::endl;
    std::cout << "  Max |a_y|:     " << max_a_y << " m/s²" << std::endl;
    std::cout << "  Max |pitch|:   " << max_pitch << "°" << std::endl;
    std::cout << "  Max |roll|:    " << max_roll << "°" << std::endl;
    std::cout << "\n✓ With horizontal control disabled, all values should be ≈0" << std::endl;
    
    // Print sample values
    std::cout << "\nSample values at different times:" << std::endl;
    for (size_t i = 0; i < time_vec.size(); i += 100) {
        std::cout << "  t=" << time_vec[i] << "s: a_x=" << a_x_vec[i] 
                  << ", a_y=" << a_y_vec[i] 
                  << ", pitch=" << pitch_des_vec[i] << "°, roll=" << roll_des_vec[i] << "°" << std::endl;
    }
}

void test_with_inputs(double current_roll, double current_pitch, double current_yaw,
                     double a_world_x, double a_world_y, double a_world_z,
                     double a_prev_x, double a_prev_y, double max_xy_acc_step) {
    
    print_separator("FORWARD CALCULATION TEST");
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== INPUTS ===" << std::endl;
    std::cout << "Current Attitude:" << std::endl;
    std::cout << "  roll  = " << std::setw(8) << current_roll*180/M_PI << "° (" << current_roll << " rad)" << std::endl;
    std::cout << "  pitch = " << std::setw(8) << current_pitch*180/M_PI << "° (" << current_pitch << " rad)" << std::endl;
    std::cout << "  yaw   = " << std::setw(8) << current_yaw*180/M_PI << "° (" << current_yaw << " rad)" << std::endl;
    
    std::cout << "\nDesired World Acceleration:" << std::endl;
    std::cout << "  a_world_x = " << std::setw(8) << a_world_x << " m/s²" << std::endl;
    std::cout << "  a_world_y = " << std::setw(8) << a_world_y << " m/s²" << std::endl;
    std::cout << "  a_world_z = " << std::setw(8) << a_world_z << " m/s²" << std::endl;
    
    std::cout << "\nPrevious Body Acceleration (for jerk limiting):" << std::endl;
    std::cout << "  a_prev_x = " << std::setw(8) << a_prev_x << " m/s²" << std::endl;
    std::cout << "  a_prev_y = " << std::setw(8) << a_prev_y << " m/s²" << std::endl;
    std::cout << "  max_xy_acc_step = " << max_xy_acc_step << " m/s²" << std::endl;
    
    // ===== CALCULATION STARTS (EXACT CODE FROM YOUR CONTROLLER) =====
    
    // Step 1: Create quaternion and rotation matrices
    Eigen::Quaterniond q = euler_to_quaternion(current_roll, current_pitch, current_yaw);
    Eigen::Matrix3d R_wb = q.toRotationMatrix();   // body -> world
    Eigen::Matrix3d R_bw = R_wb.transpose();       // world -> body
    
    // Step 2: Create world acceleration vector
    Eigen::Vector3d a_des_world(a_world_x, a_world_y, a_world_z);
    
    // Step 3: Transform to body frame
    Eigen::Vector3d a_des_body = R_bw * a_des_world;
    
    // Step 4: Jerk / slew limit on XY accelerations
    Eigen::Vector2d a_des_prev_xy(a_prev_x, a_prev_y);
    Eigen::Vector2d a_des_body_xy(a_des_body(0), a_des_body(1));
    
    std::cout << "\n=== INTERMEDIATE CALCULATIONS ===" << std::endl;
    std::cout << "After world->body transformation (before jerk limit):" << std::endl;
    std::cout << "  a_body_x = " << std::setw(8) << a_des_body(0) << " m/s²" << std::endl;
    std::cout << "  a_body_y = " << std::setw(8) << a_des_body(1) << " m/s²" << std::endl;
    std::cout << "  a_body_z = " << std::setw(8) << a_des_body(2) << " m/s²" << std::endl;
    
    std::cout << "\nJerk limiting:" << std::endl;
    for (int i = 0; i < 2; ++i) {
        double delta = a_des_body_xy(i) - a_des_prev_xy(i);
        double delta_orig = delta;
        limit(delta, -max_xy_acc_step, max_xy_acc_step);
        a_des_body_xy(i) = a_des_prev_xy(i) + delta;
        
        std::cout << "  Axis " << i << ": delta=" << std::setw(8) << delta_orig 
                  << " -> limited=" << std::setw(8) << delta 
                  << " -> new_accel=" << std::setw(8) << a_des_body_xy(i) << std::endl;
    }
    
    // Step 5: Store final body accelerations (this is a_des_)
    Eigen::Vector3d a_des;
    a_des(0) = a_des_body_xy(0);
    a_des(1) = a_des_body_xy(1);
    a_des(2) = a_des_body(2);
    
    // Step 6: Calculate desired attitude from body accelerations
    double pitch_des = -atan2(a_des(0), sqrt(a_des(2)*a_des(2) + a_des(1)*a_des(1)));
    double roll_des = atan2(a_des(1), std::abs(a_des(2)));
    
    // ===== OUTPUT RESULTS =====
    
    std::cout << "\n=== OUTPUTS (a_des_ and att_des) ===" << std::endl;
    std::cout << "Final Body Acceleration (a_des_):" << std::endl;
    std::cout << "  a_des_(0) = " << std::setw(8) << a_des(0) << " m/s²" << std::endl;
    std::cout << "  a_des_(1) = " << std::setw(8) << a_des(1) << " m/s²" << std::endl;
    std::cout << "  a_des_(2) = " << std::setw(8) << a_des(2) << " m/s²" << std::endl;
    
    std::cout << "\nDesired Attitude (from a_des_):" << std::endl;
    std::cout << "  pitch_des = " << std::setw(8) << pitch_des*180/M_PI << "° (" << pitch_des << " rad)" << std::endl;
    std::cout << "  roll_des  = " << std::setw(8) << roll_des*180/M_PI << "° (" << roll_des << " rad)" << std::endl;
    
    std::cout << "\n=== ROTATION MATRICES ===" << std::endl;
    std::cout << "R_wb (body->world):" << std::endl << R_wb << std::endl;
    std::cout << "\nR_bw (world->body):" << std::endl << R_bw << std::endl;
    
    print_separator();
}

int main(int argc, char** argv) {
    
    // Check if user provided custom inputs
    if (argc >= 10) {
        // Parse command line arguments
        double roll = std::stod(argv[1]);
        double pitch = std::stod(argv[2]);
        double yaw = std::stod(argv[3]);
        double a_wx = std::stod(argv[4]);
        double a_wy = std::stod(argv[5]);
        double a_wz = std::stod(argv[6]);
        double a_px = std::stod(argv[7]);
        double a_py = std::stod(argv[8]);
        double max_step = std::stod(argv[9]);
        
        test_with_inputs(roll, pitch, yaw, a_wx, a_wy, a_wz, a_px, a_py, max_step);
        
    } else {
        // Run preset test cases
        print_separator("CONTROL MATH TEST SUITE");
        std::cout << "Testing with preset scenarios" << std::endl;
        std::cout << "Usage: ./test_control_math roll pitch yaw a_wx a_wy a_wz a_px a_py max_step" << std::endl;
        std::cout << "  (angles in radians, accelerations in m/s²)" << std::endl;
        print_separator();
        
        std::cout << "\n\nTEST CASE 1: Hover with horizontal control disabled\n" << std::endl;
        test_with_inputs(
            0.0, 0.0, 0.0,           // current attitude (hover)
            0.0, 0.0, -9.81,         // world accel (no horizontal)
            0.0, 0.0,                // previous body accel
            2.0                      // max step
        );
        
        std::cout << "\n\nTEST CASE 2: Forward acceleration command\n" << std::endl;
        test_with_inputs(
            0.0, 0.0, 0.0,           // current attitude (hover)
            -3.0, 0.0, -9.81,        // world accel (forward command)
            0.0, 0.0,                // previous body accel
            2.0                      // max step
        );
        
        std::cout << "\n\nTEST CASE 3: Already tilted, adding right command\n" << std::endl;
        test_with_inputs(
            0.0, 0.1745, 0.0,        // current attitude (pitched 10°)
            0.0, 2.0, -9.81,         // world accel (right command)
            -1.5, 0.0,               // previous body accel
            2.0                      // max step
        );
        
        std::cout << "\n\nTEST CASE 4: Large acceleration change (tests jerk limiting)\n" << std::endl;
        test_with_inputs(
            0.0, 0.0, 0.0,           // current attitude (hover)
            5.0, 5.0, -9.81,         // world accel (large command)
            0.0, 0.0,                // previous body accel
            2.0                      // max step (will limit)
        );
        
        std::cout << "\n\nTEST CASE 5: Yawed attitude (tests rotation)\n" << std::endl;
        test_with_inputs(
            0.0, 0.0, 0.7854,        // current attitude (yawed 45°)
            2.0, 0.0, -9.81,         // world accel (forward in world frame)
            0.0, 0.0,                // previous body accel
            2.0                      // max step
        );
        
        print_separator("\nALL TESTS COMPLETED");
        std::cout << "\nTo test custom inputs, run:" << std::endl;
        std::cout << "  ./test_control_math roll pitch yaw a_wx a_wy a_wz a_px a_py max_step" << std::endl;
        std::cout << "\nExample (hover with no horizontal control):" << std::endl;
        std::cout << "  ./test_control_math 0 0 0 0 0 -9.81 0 0 2.0" << std::endl;
        print_separator();
    }
    
    return 0;
}
