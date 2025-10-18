#!/usr/bin/env python3
"""
Yaw Feedforward Auto-Tuner using Bayesian Optimization
Optimizes yaw_ff_dt_base and yaw_ff_dt_gain for improved ATTITUDE CONTROL
Focuses on minimizing roll/pitch errors and yaw tracking, especially during high yaw rates
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
import json
import os
import subprocess
import time
import signal
import sys
import argparse
from typing import Dict, List, Tuple, Optional

class YawFFAutoTuner:
    """Auto-tuner specifically for yaw feedforward parameters"""
    
    # Define search space for yaw feedforward parameters
    search_space = [
        Real(0.000, 0.060, name='yaw_ff_dt_base'),    # Base prediction horizon (20-60ms)
        Real(0.000, 0.020, name='yaw_ff_dt_gain'),    # Additional dt per rad/s (0-20ms)
        Real(0.050, 0.150, name='yaw_ff_dt_max'),     # Maximum prediction horizon (50-150ms)
        Real(0.05, 0.40, name='yaw_rate_lpf_alpha'),  # LPF alpha for gyro yaw rate
    ]

    @staticmethod
    def _canon_key(params: List[float]) -> Tuple[float, ...]:
        """Round parameters for stable hashing/deduplication."""
        return tuple(round(float(x), 6) for x in params)

    def _names(self) -> List[str]:
        return [d.name for d in self.search_space]

    def _dict_to_vector(self, params_dict: Dict[str, float]) -> List[float]:
        names = self._names()
        return [float(params_dict[name]) for name in names if name in params_dict]

    def _vector_to_dict(self, params_vec: List[float]) -> Dict[str, float]:
        return dict(zip(self._names(), params_vec))

    def __init__(self, config_file: str, flight_data_dir: str,
                 reset_history: Optional[bool] = None,
                 max_seed_cost: Optional[float] = None,
                 results_file: Optional[str] = None,
                 n_repeats: int = 3):
        self.config_file = config_file
        self.flight_data_dir = flight_data_dir
        self.base_config = self.load_config()
        
        # Performance tracking
        self.trial_results = []
        self.best_params = None
        self.best_cost = float('inf')
        self.n_repeats = n_repeats  # Number of repeated runs per parameter set
        
        # Graceful shutdown control
        self.stop_requested = False
        self.in_trial = False
        self._signals_received = 0
        
        # Results database
        self.results_output_dir = os.path.join(self.flight_data_dir, 'outputs')
        os.makedirs(self.results_output_dir, exist_ok=True)
        self.results_file = results_file or os.path.join(self.results_output_dir, 'yaw_ff_tuning_results.json')
        
        # Cache of evaluated params -> cost
        self.evaluated_cache: Dict[Tuple[float, ...], float] = {}
        
        # Seed data for optimizer
        self._seed_x0: List[List[float]] = []
        self._seed_y0: List[float] = []
        
        # Behavior controls
        self.reset_history = (
            reset_history if reset_history is not None
            else os.environ.get('AUTOTUNE_RESET_HISTORY', '').lower() in ('1', 'true', 'yes')
        )
        self.max_seed_cost = (
            float(max_seed_cost) if max_seed_cost is not None
            else float(os.environ.get('AUTOTUNE_MAX_SEED_COST', '30'))
        )
        
        # Load existing results
        self._load_previous_results()
        
        # Warm start with historical best
        if self.best_params is not None:
            try:
                self.update_config(self.best_params)
                print(f"Warm-starting from historical best (cost={self.best_cost:.4f})")
            except Exception as e:
                print(f"Warning: Could not apply historical best params: {e}")

    def install_signal_handlers(self):
        """Install handlers for graceful shutdown"""
        def _handler(sig, frame):
            self._signals_received += 1
            name = signal.Signals(sig).name
            if self._signals_received == 1:
                self.stop_requested = True
                print(f"\nReceived {name}. Will stop after this trial completes...", flush=True)
            else:
                print(f"\nReceived {name} again. Forcing immediate exit...", flush=True)
                raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def load_config(self) -> Dict:
        """Load current configuration"""
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def update_config(self, params: List[float]) -> None:
        """Update configuration with new yaw feedforward parameters
        
        Note: This assumes the C++ code reads from a config file.
        You'll need to add yaw_ff parameters to your JSON config structure.
        """
        config = self.base_config.copy()
        
        # Add yaw_ff section if it doesn't exist
        if 'yaw_ff' not in config:
            config['yaw_ff'] = {}
        
        # Map optimization parameters to config structure
        config['yaw_ff']['dt_base'] = params[0]      # yaw_ff_dt_base
        config['yaw_ff']['dt_gain'] = params[1]      # yaw_ff_dt_gain
        config['yaw_ff']['dt_max'] = params[2]       # yaw_ff_dt_max
        config['yaw_ff']['lpf_alpha'] = params[3]    # yaw_rate_lpf_alpha
        
        # Write updated config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Updated config: base={params[0]:.4f}s, gain={params[1]:.4f}s/rad, max={params[2]:.4f}s, alpha={params[3]:.3f}")

    def _load_previous_results(self) -> None:
        """Load previous tuning results to seed optimizer"""
        try:
            if self.reset_history or not os.path.isfile(self.results_file):
                print("History disabled or no results DB found; starting fresh.")
                return
            
            with open(self.results_file, 'r') as f:
                db = json.load(f)
            
            trials = db.get('trial_history', []) or []
            names = self._names()
            
            for t in trials:
                if t is None or t.get('failed'):
                    continue
                params_dict = t.get('params')
                cost = t.get('cost')
                if params_dict is None or cost is None:
                    continue
                if not all(n in params_dict for n in names):
                    continue
                
                vec = self._dict_to_vector(params_dict)
                key = self._canon_key(vec)
                
                try:
                    c = float(cost)
                except Exception:
                    continue
                
                # Update cache
                prev_cost = self.evaluated_cache.get(key)
                if prev_cost is None or c < prev_cost:
                    self.evaluated_cache[key] = c
                
                # Seed optimizer if within threshold
                if c <= self.max_seed_cost:
                    self._seed_x0.append(vec)
                    self._seed_y0.append(c)
            
            # Initialize best from DB
            db_best_cost = db.get('best_cost')
            db_best_params = db.get('best_params')
            if db_best_cost is not None and db_best_params is not None:
                if all(n in db_best_params for n in names):
                    self.best_cost = float(db_best_cost)
                    self.best_params = self._dict_to_vector(db_best_params)
            
            print(f"History loaded: {len(self._seed_x0)} seed points from {len(trials)} prior trials")
        except Exception as e:
            print(f"Warning: Could not load previous results: {e}")

    def cleanup_all_processes(self):
        """Comprehensive cleanup of all PX4/Gazebo processes"""
        print("Performing comprehensive process cleanup...")
        
        processes_to_kill = [
            "px4", "gz", "gzserver", "gzclient", "gazebo",
            "offboard_control_spin_tt"
        ]
        
        for attempt in range(3):
            for process in processes_to_kill:
                os.system(f"pkill -9 {process} >/dev/null 2>&1 || true")
            time.sleep(1)
        
        print("Cleanup complete.")

    def wait_for_px4_sitl_ready(self, timeout=60):
        """Wait for PX4 SITL to be ready"""
        print("Waiting for PX4 SITL to be ready...")
        start = time.time()
        ready = False
        
        while time.time() - start < timeout:
            try:
                with open("/tmp/px4_sitl_gz.log", "r") as f:
                    log = f.read()
                    if ("INFO  [uxrce_dds_client] successfully created rt/fmu/out/" in log or
                        "INFO  [uxrce_dds_client] synchronized with time offset" in log):
                        ready = True
                        break
            except Exception:
                pass
            
            time.sleep(2)
        
        if ready:
            time.sleep(15)
            print("PX4 SITL is ready!")
        else:
            print("Warning: PX4 SITL may not be fully ready after timeout!")
            time.sleep(10)

    def run_flight_test(self, duration: float = 22.0) -> str:
        """Run a flight test and return log filename"""
        print(f"Starting flight test with duration {duration}s...")
        
        # Kill previous instances
        kill_patterns = ["px4", "gz", "gzserver", "gzclient", "gazebo", "offboard_control_spin_tt"]
        for pat in kill_patterns:
            subprocess.run(f"pkill -f {pat}", shell=True, stderr=subprocess.DEVNULL)
        time.sleep(2)
        
        # Launch PX4 SITL + Gazebo
        print("Launching PX4 SITL + Gazebo...")
        px4 = subprocess.Popen(
            "HEADLESS=1 make px4_sitl gz_x500 > /tmp/px4_sitl_gz.log 2>&1",
            cwd="/home/pyro/PX4-Autopilot",
            shell=True,
            executable="/bin/bash",
        )
        
        self.wait_for_px4_sitl_ready(timeout=60)
        
        # Launch controller
        print("Launching controller node...")
        start_time = time.time()
        ctrl = subprocess.Popen(
            "source install/local_setup.bash && ros2 run px4_ros_com offboard_control_spin_tt",
            cwd="/home/pyro/ws_offboard_control",
            shell=True,
            executable="/bin/bash",
        )
        
        # Let it fly
        time.sleep(max(0.0, duration))
        
        # Wait for controller to finish
        print("Waiting for controller to finish and save logs...")
        wait_deadline = time.time() + 15.0
        while time.time() < wait_deadline:
            if ctrl.poll() is not None:
                break
            time.sleep(0.5)
        
        if ctrl.poll() is None:
            try:
                ctrl.send_signal(signal.SIGINT)
                ctrl.wait(timeout=8)
            except Exception:
                pass
        
        if ctrl.poll() is None:
            try:
                ctrl.terminate()
                ctrl.wait(timeout=5)
            except Exception:
                ctrl.kill()
        
        # Terminate PX4
        try:
            px4.terminate()
            px4.wait(timeout=5)
        except Exception:
            px4.kill()
        
        # Find log file
        log_dir = f"{self.flight_data_dir}/logs"
        os.makedirs(log_dir, exist_ok=True)
        print("Waiting for flight CSV log to appear...")
        log_path = None
        deadline = time.time() + 30.0
        
        while time.time() < deadline and log_path is None:
            try:
                candidates = []
                for f in os.listdir(log_dir):
                    if f.endswith('.csv'):
                        full = os.path.join(log_dir, f)
                        try:
                            if os.path.getmtime(full) >= start_time:
                                candidates.append(full)
                        except FileNotFoundError:
                            continue
                if candidates:
                    log_path = max(candidates, key=os.path.getmtime)
                    if os.path.getsize(log_path) < 100:
                        log_path = None
                if log_path is None:
                    time.sleep(0.5)
            except FileNotFoundError:
                time.sleep(0.5)
        
        if log_path is None:
            raise RuntimeError("No log file generated (timeout waiting for CSV)")
        
        return log_path

    def check_takeoff(self, log_file: str) -> bool:
        """Check if drone successfully took off within first 4 seconds"""
        try:
            import pandas as pd
            df = pd.read_csv(log_file)
            
            if len(df) < 10:
                return False
            
            # Get first 4 seconds of data
            time = df['timestamp'].to_numpy()
            start_time = time[0]
            takeoff_window = (time - start_time) <= 4.0
            
            if not np.any(takeoff_window):
                return False
            
            # Check altitude during takeoff window
            z_pos = df['pos_z'].to_numpy()
            z_in_window = z_pos[takeoff_window]
            
            # Consider successful takeoff if altitude reaches -5m (5m up) within 4 seconds
            # (z is negative up in NED frame)
            if np.any(z_in_window < -5.0):
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking takeoff: {e}")
            return False

    def analyze_performance(self, log_file: str) -> float:
        """Analyze flight performance focusing on ATTITUDE ERROR
        
        Cost function emphasizes:
        1. Roll and pitch attitude errors (primary)
        2. Yaw angle tracking
        3. Overall attitude stability
        4. Control smoothness
        """
        try:
            import pandas as pd
            df = pd.read_csv(log_file)
            
            if len(df) < 10:
                return 1000.0
            
            # Extract relevant columns for ATTITUDE ERROR
            time = df['timestamp'].to_numpy()
            
            # Attitude errors (PRIMARY FOCUS)
            roll = df['roll'].to_numpy()
            pitch = df['pitch'].to_numpy()
            yaw = df['yaw'].to_numpy()
            roll_des = df['roll_des'].to_numpy()
            pitch_des = df['pitch_des'].to_numpy()
            yaw_des = df['yaw_des'].to_numpy()
            
            # Compute attitude errors
            roll_err = roll_des - roll
            pitch_err = pitch_des - pitch
            yaw_err = yaw_des - yaw
            # Wrap yaw error to [-pi, pi]
            yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))
            
            # Rate errors (secondary)
            p_err = df['rate_err_p'].to_numpy()
            q_err = df['rate_err_q'].to_numpy()
            r_err = df['rate_err_r'].to_numpy()
            
            # Control commands
            torque_x = df['torque_x'].to_numpy()
            torque_y = df['torque_y'].to_numpy()
            torque_z = df['torque_z'].to_numpy()
            
            # 1. ROLL & PITCH ATTITUDE ERROR (MOST IMPORTANT)
            roll_rms = np.sqrt(np.mean(roll_err**2))
            pitch_rms = np.sqrt(np.mean(pitch_err**2))
            attitude_rms = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
            
            # 2. Peak attitude errors (penalize large deviations)
            roll_peak = np.max(np.abs(roll_err))
            pitch_peak = np.max(np.abs(pitch_err))
            peak_penalty = roll_peak + pitch_peak
            
            # 3. YAW ANGLE ERROR (feedforward affects this)
            yaw_rms = np.sqrt(np.mean(yaw_err**2))
            
            # 4. Attitude error during high yaw rates (feedforward effectiveness)
            r = df['r'].to_numpy()
            yaw_rate_mag = np.abs(r)
            high_rate_mask = yaw_rate_mag > 2.0  # rad/s threshold
            if np.sum(high_rate_mask) > 10:
                # Roll/pitch errors during high yaw rates (coupling effects)
                roll_err_high_rate = np.sqrt(np.mean(roll_err[high_rate_mask]**2))
                pitch_err_high_rate = np.sqrt(np.mean(pitch_err[high_rate_mask]**2))
                yaw_err_high_rate = np.sqrt(np.mean(yaw_err[high_rate_mask]**2))
            else:
                roll_err_high_rate = 0.0
                pitch_err_high_rate = 0.0
                yaw_err_high_rate = 0.0
            
            # 5. Rate tracking errors (secondary importance)
            rate_rms = np.sqrt(np.mean(p_err**2 + q_err**2 + r_err**2))
            
            # 6. Control smoothness (avoid oscillations)
            torque_smoothness = np.mean(
                np.diff(torque_x)**2 + 
                np.diff(torque_y)**2 + 
                np.diff(torque_z)**2
            )
            
            # 7. Attitude stability in steady state (last 25% of flight)
            steady_idx = int(len(df) * 0.75)
            if len(df) - steady_idx > 10:
                steady_roll_std = np.std(roll_err[steady_idx:])
                steady_pitch_std = np.std(pitch_err[steady_idx:])
                stability_penalty = steady_roll_std + steady_pitch_std
            else:
                stability_penalty = 0.0
            
            # WEIGHTED COST FUNCTION - FOCUSED ON ATTITUDE ERROR
            cost = (
                # Primary: Roll & Pitch attitude errors
                20.0 * roll_rms +                    # Roll RMS error (highest weight)
                20.0 * pitch_rms +                   # Pitch RMS error (highest weight)
                15.0 * attitude_rms +                # Combined attitude RMS
                10.0 * peak_penalty +                # Peak attitude deviations
                
                # Secondary: Yaw angle tracking (feedforward effectiveness)
                8.0 * yaw_rms +                      # Yaw angle RMS error
                12.0 * yaw_err_high_rate +          # Yaw error during high rates
                
                # Coupling effects during yaw maneuvers
                6.0 * roll_err_high_rate +          # Roll coupling during yaw
                6.0 * pitch_err_high_rate +         # Pitch coupling during yaw
                
                # Tertiary: Rate tracking and smoothness
                2.0 * rate_rms +                    # Rate tracking errors
                1.0 * torque_smoothness +           # Control smoothness
                5.0 * stability_penalty             # Steady-state stability
            )
            
            print(f"  Attitude Performance:")
            print(f"    Roll RMS: {roll_rms:.4f} rad ({roll_rms*57.3:.2f}°), Peak: {roll_peak*57.3:.2f}°")
            print(f"    Pitch RMS: {pitch_rms:.4f} rad ({pitch_rms*57.3:.2f}°), Peak: {pitch_peak*57.3:.2f}°")
            print(f"    Yaw RMS: {yaw_rms:.4f} rad ({yaw_rms*57.3:.2f}°)")
            print(f"    High-rate yaw err: {yaw_err_high_rate:.4f} rad ({yaw_err_high_rate*57.3:.2f}°)")
            print(f"    Roll/Pitch coupling: {roll_err_high_rate:.4f}/{pitch_err_high_rate:.4f} rad")
            print(f"    Smoothness: {torque_smoothness:.4f}, Stability: {stability_penalty:.4f}")
            
            return float(cost)
            
        except Exception as e:
            print(f"Error analyzing performance: {e}")
            import traceback
            traceback.print_exc()
            return 1000.0

    def objective_function(self, param_values) -> float:
        """Objective function for Bayesian optimization
        Runs multiple trials for each parameter set and returns average cost"""
        if self.stop_requested:
            print("Stop requested before starting new trial. Aborting...", flush=True)
            raise KeyboardInterrupt
        
        trial_idx = len(self.trial_results) + 1
        print(f"\n{'='*60}")
        print(f"Trial {trial_idx} (will run {self.n_repeats} times for averaging)")
        print(f"Testing parameters: {dict(zip([d.name for d in self.search_space], param_values))}")
        print(f"{'='*60}")
        sys.stdout.flush()
        
        # Check cache
        key = self._canon_key(param_values)
        if key in self.evaluated_cache:
            cached_cost = self.evaluated_cache[key]
            print(f"Parameters previously evaluated. Using cached cost: {cached_cost:.4f}")
            result = {
                'params': self._vector_to_dict(param_values),
                'cost': cached_cost,
                'log_file': None,
                'cached': True
            }
            self.trial_results.append(result)
            if cached_cost < self.best_cost:
                self.best_cost = cached_cost
                self.best_params = list(param_values)
                print(f"New best cost (from cache): {cached_cost:.4f}")
            return cached_cost
        
        # Update configuration
        self.update_config(param_values)
        
        # Run multiple flight tests and collect costs
        self.in_trial = True
        costs = []
        log_files = []
        max_retries = 2  # Retry failed runs up to 2 times
        
        try:
            for repeat_idx in range(self.n_repeats):
                print(f"\n  --- Run {repeat_idx + 1}/{self.n_repeats} ---")
                
                success = False
                retry_count = 0
                
                while not success and retry_count <= max_retries:
                    if retry_count > 0:
                        print(f"  Retry attempt {retry_count}/{max_retries}...")
                    
                    try:
                        log_file = self.run_flight_test(duration=20.0)
                        
                        # Check if takeoff succeeded
                        if not self.check_takeoff(log_file):
                            print(f"  ⚠️  Takeoff failed - drone didn't reach 5m altitude in 4 seconds")
                            retry_count += 1
                            if retry_count > max_retries:
                                print(f"  ❌ Max retries reached. Recording as failed run.")
                                costs.append(1000.0)
                                log_files.append(None)
                            continue
                        
                        # Takeoff succeeded, analyze performance
                        cost = self.analyze_performance(log_file)
                        costs.append(cost)
                        log_files.append(log_file)
                        print(f"  ✓ Run {repeat_idx + 1} cost: {cost:.4f}")
                        success = True
                        
                    except Exception as e:
                        print(f"  ⚠️  Run failed with error: {e}")
                        retry_count += 1
                        if retry_count > max_retries:
                            print(f"  ❌ Max retries reached. Recording as failed run.")
                            costs.append(1000.0)
                            log_files.append(None)
                
                # Check for stop request between runs
                if self.stop_requested:
                    print("Stop flag detected between runs. Using partial results.", flush=True)
                    break
            
            # Calculate statistics
            if len(costs) == 0:
                raise RuntimeError("All runs failed")
            
            valid_costs = [c for c in costs if c < 999.0]  # Exclude failure costs
            
            if len(valid_costs) == 0:
                # All runs failed
                avg_cost = 1000.0
                std_cost = 0.0
                print(f"\n  ❌ All runs failed. Using penalty cost: {avg_cost:.4f}")
            else:
                avg_cost = float(np.mean(valid_costs))
                std_cost = float(np.std(valid_costs)) if len(valid_costs) > 1 else 0.0
                min_cost = float(np.min(valid_costs))
                max_cost = float(np.max(valid_costs))
                
                print(f"\n  📊 Statistics over {len(valid_costs)} valid run(s):")
                print(f"     Average: {avg_cost:.4f}")
                print(f"     Std Dev: {std_cost:.4f}")
                print(f"     Min: {min_cost:.4f}")
                print(f"     Max: {max_cost:.4f}")
                print(f"     Range: {max_cost - min_cost:.4f}")
            
            # Track results with all run information
            result = {
                'params': self._vector_to_dict(param_values),
                'cost': avg_cost,
                'cost_std': std_cost,
                'costs_all_runs': costs,
                'n_valid_runs': len(valid_costs),
                'n_total_runs': len(costs),
                'log_files': log_files
            }
            self.trial_results.append(result)
            
            # Update cache with average cost
            self.evaluated_cache[key] = float(avg_cost)
            
            # Update best based on average cost
            if avg_cost < self.best_cost:
                self.best_cost = avg_cost
                self.best_params = param_values.copy()
                print(f"\n{'*'*60}")
                print(f"NEW BEST AVERAGE COST: {avg_cost:.4f} (±{std_cost:.4f})")
                print(f"Parameters: {self._vector_to_dict(param_values)}")
                print(f"{'*'*60}\n")
            else:
                print(f"\nAverage Cost: {avg_cost:.4f} (±{std_cost:.4f})")
            
            sys.stdout.flush()
            
            if self.stop_requested:
                print("Stop flag detected. Exiting after printing cost.", flush=True)
                raise KeyboardInterrupt
            
            return avg_cost
            
        except Exception as e:
            avg_cost = 1000.0
            print(f"\nTrial failed catastrophically: {e}")
            print(f"Average Cost: {avg_cost:.4f} (failure)")
            sys.stdout.flush()
            
            self.trial_results.append({
                'params': self._vector_to_dict(param_values),
                'cost': avg_cost,
                'cost_std': 0.0,
                'costs_all_runs': costs if costs else [1000.0],
                'n_valid_runs': 0,
                'n_total_runs': len(costs) if costs else 0,
                'log_files': log_files if log_files else [],
                'failed': True,
                'error': str(e)
            })
            self.evaluated_cache[key] = float(avg_cost)
            
            if self.stop_requested:
                raise KeyboardInterrupt
            
            return avg_cost
        finally:
            self.in_trial = False

    def optimize(self, n_calls: int = 30) -> Dict:
        """Run Bayesian optimization"""
        print(f"\n{'='*70}")
        print(f"Starting Yaw Feedforward Bayesian Optimization")
        print(f"Target: {n_calls} trials")
        print(f"{'='*70}\n")
        
        try:
            # Prepare seed data
            x0: List[List[float]] = []
            y0: List[float] = []
            seen_keys: Dict[Tuple[float, ...], float] = {}
            
            for x, y in zip(self._seed_x0, self._seed_y0):
                k = self._canon_key(x)
                prev = seen_keys.get(k)
                if prev is None or y < prev:
                    seen_keys[k] = float(y)
            
            if seen_keys:
                for k, y in seen_keys.items():
                    x0.append(list(k))
                    y0.append(float(y))
            
            n_init_default = 5  # Fewer initial points for yaw FF tuning
            n_init = max(0, n_init_default - len(x0)) if x0 else n_init_default
            
            result = gp_minimize(
                func=self.objective_function,
                dimensions=self.search_space,
                x0=x0 if x0 else None,
                y0=y0 if x0 else None,
                n_calls=n_calls,
                n_initial_points=n_init,
                acq_func='gp_hedge',
                acq_optimizer='sampling',
                random_state=42,
                verbose=True
            )
            
            # Update config with best parameters
            if self.best_params is not None:
                self.update_config(self.best_params)
                print(f"\n{'='*70}")
                print(f"OPTIMIZATION COMPLETE!")
                print(f"{'='*70}")
                print(f"Best cost: {self.best_cost:.4f}")
                print(f"Best parameters:")
                for name, value in zip([d.name for d in self.search_space], self.best_params):
                    print(f"  {name}: {value:.6f}")
                print(f"{'='*70}\n")
            
            return {
                'best_params': self.best_params,
                'best_cost': self.best_cost,
                'all_results': self.trial_results,
                'optimization_result': result
            }
            
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user")
            return {
                'best_params': self.best_params,
                'best_cost': self.best_cost,
                'all_results': self.trial_results,
                'optimization_result': None
            }
        finally:
            print("\nPerforming final cleanup...")
            self.cleanup_all_processes()

    def save_results(self, results: Dict, filename: str) -> None:
        """Save optimization results with deduplication"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        names = self._names()
        
        # Load existing DB
        existing = {}
        if os.path.isfile(filename):
            try:
                with open(filename, 'r') as f:
                    existing = json.load(f) or {}
            except Exception as e:
                print(f"Warning: Could not read existing results: {e}")
        
        existing_trials = existing.get('trial_history', []) or []
        new_trials = results.get('all_results', []) or []
        combined: Dict[Tuple[float, ...], Dict] = {}
        
        def to_key_from_trial(tr: Dict) -> Tuple[float, ...]:
            pd = tr.get('params') or {}
            if not all(n in pd for n in names):
                return tuple()
            vec = [float(pd[n]) for n in names]
            return self._canon_key(vec)
        
        # Merge trials
        for tr in existing_trials + new_trials:
            key = to_key_from_trial(tr)
            if key == tuple():
                continue
            cost = tr.get('cost')
            if key not in combined:
                combined[key] = tr
            else:
                try:
                    if cost is not None and combined[key].get('cost') is not None:
                        if float(cost) < float(combined[key]['cost']):
                            combined[key] = tr
                except Exception:
                    pass
        
        merged_trials = list(combined.values())
        
        # Recompute best
        best_cost = None
        best_params_dict = None
        for tr in merged_trials:
            if tr.get('failed'):
                continue
            c = tr.get('cost')
            pd = tr.get('params')
            if c is None or pd is None:
                continue
            try:
                c = float(c)
            except Exception:
                continue
            if best_cost is None or c < best_cost:
                best_cost = c
                best_params_dict = {n: float(pd[n]) for n in names if n in pd}
        
        output = {
            'best_cost': best_cost if best_cost is not None else results.get('best_cost'),
            'best_params': best_params_dict if best_params_dict is not None else (
                dict(zip(names, results.get('best_params'))) if results.get('best_params') else existing.get('best_params')
            ),
            'trial_history': merged_trials
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    """Run yaw feedforward automated tuning"""
    parser = argparse.ArgumentParser(description="Yaw Feedforward Auto-Tuner")
    parser.add_argument("--use-history", dest="use_history", action="store_true", 
                       help="Use prior tuning results to warm start", default=False)
    parser.add_argument("--no-history", dest="use_history", action="store_false",
                       help="Ignore history and start fresh (default)")
    parser.add_argument("--max-seed-cost", type=float, default=None,
                       help="Max cost from history to seed the optimizer")
    parser.add_argument("--results-file", type=str, default=None,
                       help="Path to results DB JSON (default: yaw_ff_tuning_results.json)")
    parser.add_argument("--trials", type=int, default=30,
                       help="Number of optimization calls")
    parser.add_argument("--repeats", type=int, default=3,
                       help="Number of repeated runs per parameter set (default: 3)")
    args = parser.parse_args()
    
    config_file = "/home/pyro/ws_offboard_control/flight_data/config/control_params.json"
    flight_data_dir = "/home/pyro/ws_offboard_control/flight_data"
    
    # Use separate results file by default (yaw_ff_tuning_results.json vs tuning_results.json)
    # This ensures yaw FF tuning doesn't interfere with main controller tuning
    tuner = YawFFAutoTuner(
        config_file,
        flight_data_dir,
        reset_history=(not args.use_history),
        max_seed_cost=args.max_seed_cost,
        results_file=args.results_file,  # Defaults to yaw_ff_tuning_results.json in __init__
        n_repeats=args.repeats
    )
    
    tuner.install_signal_handlers()
    
    # Print configuration summary
    print(f"\n{'='*70}")
    print(f"TUNER CONFIGURATION")
    print(f"{'='*70}")
    print(f"Config file: {config_file}")
    print(f"Results file: {tuner.results_file}")
    print(f"Use history: {args.use_history}")
    print(f"Reset history: {not args.use_history}")
    print(f"Number of trials: {args.trials}")
    print(f"Repeats per trial: {args.repeats} (averaging for robustness)")
    print(f"Total flights: {args.trials * args.repeats}")
    if tuner.best_params is not None:
        print(f"Starting from historical best: cost={tuner.best_cost:.4f}")
    else:
        print(f"Starting fresh (no historical best)")
    print(f"{'='*70}\n")
    
    # Run optimization
    results = tuner.optimize(n_calls=args.trials)
    
    # Save results
    out_file = args.results_file or f"{flight_data_dir}/outputs/yaw_ff_tuning_results.json"
    tuner.save_results(results, out_file)

if __name__ == "__main__":
    main()
