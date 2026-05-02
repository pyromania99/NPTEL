#!/usr/bin/env python3
"""
Automated Controller Tuning using Bayesian Optimization
Optimizes NDI controller gains based on flight performance metrics
"""

import numpy as np
import sys
from skopt import gp_minimize
from skopt.space import Real
import json
import os
import subprocess
import time
from typing import Dict, List, Tuple, Optional
import signal
import argparse

class ControllerAutoTuner:
    """
    ControllerAutoTuner now supports selecting which parameters to tune via boolean flags.

    How to choose parameters to tune (any of the following):
    - Edit flight_data/config/tune_flags.json and set each parameter to true/false
      Example:
      {
        "ndi_kp": true,
        "ndi_kd": false,
        "fb_kp": true,
        "fb_kd": false,
        "att_kp": true,
        "att_kd": false
      }
    - Use CLI: --tune-only att_kp,att_kd (only these will be tuned; others disabled)
    - Use CLI: --tune-except fb_kd,ndi_kd (all enabled except these)
    - Print available names: --list-params
    """
    # Helper: canonicalize parameter vectors for stable dictionary keys
    @staticmethod
    def _canon_key(params: List[float]) -> Tuple[float, ...]:
        """Round parameters for stable hashing/deduplication."""
        return tuple(round(float(x), 6) for x in params)

    def _names(self) -> List[str]:
        # Active dimension names only
        return [d.name for d in self.active_space]

    def _dict_to_vector(self, params_dict: Dict[str, float]) -> List[float]:
        names = self._names()
        return [float(params_dict[name]) for name in names if name in params_dict]

    def _vector_to_dict(self, params_vec: List[float]) -> Dict[str, float]:
        # Ensure native Python floats for JSON serialization
        names = self._names()
        return {name: float(val) for name, val in zip(names, params_vec)}

    def install_signal_handlers(self):
        """Install handlers to defer termination until after the current trial.
        - SIGINT/SIGTERM set a stop flag; we exit after printing the current trial's cost.
        - A second signal forces an immediate KeyboardInterrupt.
        """
        self._signals_received = 0

        def _handler(sig, frame):
            self._signals_received += 1
            name = signal.Signals(sig).name
            if self._signals_received == 1:
                # First signal: request graceful stop after current trial
                self.stop_requested = True
                print(f"\nReceived {name}. Will stop after this trial completes and prints the cost...", flush=True)
                # Do not raise; allow current trial to finish
            else:
                # Second signal: force immediate exit
                print(f"\nReceived {name} again. Forcing immediate exit...", flush=True)
                raise KeyboardInterrupt

        # Register handlers
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def launch_px4_sitl_gazebo(self):
        """Launch PX4 SITL with Gazebo for x500. Returns process handle."""
        print("Terminating any previous PX4/Gazebo processes...")
        
        # More comprehensive process killing
        processes_to_kill = [
            "px4",
            "gz",
            "gzserver", 
            "gzclient",
            "gazebo",
            "ninja",  # Build processes
            "make",   # Make processes that might be hanging
        ]
        
        # Kill processes multiple times to ensure they're really dead
        for attempt in range(3):
            print(f"Cleanup attempt {attempt + 1}/3...")
            for process in processes_to_kill:
                os.system(f"pkill -9 {process} >/dev/null 2>&1 || true")
                os.system(f"killall -9 {process} >/dev/null 2>&1 || true")
            
            # Also kill any processes using the PX4 ports
            os.system("lsof -ti:14540 | xargs kill -9 >/dev/null 2>&1 || true")  # PX4 SITL port
            os.system("lsof -ti:14557 | xargs kill -9 >/dev/null 2>&1 || true")  # PX4 offboard port
            
            # Wait a bit between attempts
            time.sleep(1)
        
        # Final cleanup - kill any remaining build processes in PX4-Autopilot directory
        os.system("pkill -f 'PX4-Autopilot' >/dev/null 2>&1 || true")
        os.system("pkill -f 'px4_sitl' >/dev/null 2>&1 || true")
        os.system("pkill -f 'gz_x500' >/dev/null 2>&1 || true")
        
        # Kill any cmake or make processes that might be building PX4
        os.system("pkill -f 'cmake.*px4' >/dev/null 2>&1 || true")
        os.system("pkill -f 'make.*px4' >/dev/null 2>&1 || true")
        os.system("pkill -f 'ninja.*gz' >/dev/null 2>&1 || true")
        
        # Kill any timeout commands that might be wrapping PX4 commands
        os.system("pkill -f 'timeout.*px4' >/dev/null 2>&1 || true")
        
        # Wait for processes to fully terminate
        time.sleep(3)
        
        # Verify processes are killed
        remaining_px4 = subprocess.run(["pgrep", "px4"], capture_output=True)
        remaining_gz = subprocess.run(["pgrep", "gz"], capture_output=True) 
        
        if remaining_px4.returncode == 0:
            print("Warning: Some PX4 processes may still be running")
            print("Remaining PX4 processes:", remaining_px4.stdout.decode().strip())
        
        if remaining_gz.returncode == 0:
            print("Warning: Some Gazebo processes may still be running") 
            print("Remaining Gazebo processes:", remaining_gz.stdout.decode().strip())
        
        print("Process cleanup complete.")
        
        # Start PX4 SITL with Gazebo x500 in background
        print("Starting fresh PX4 SITL + Gazebo instance (headless)...")
        # Run headless to avoid GUI overhead; supported by PX4 make for Gazebo
        sitl_cmd = "cd /home/pyro/PX4-Autopilot && HEADLESS=1 make px4_sitl gz_x500 > /tmp/px4_sitl_gz.log 2>&1 &"
        os.system(sitl_cmd)
        print("Launched PX4 SITL + Gazebo (x500)...")
        
        # Wait for PX4 SITL to be ready
        self.wait_for_px4_sitl_ready()

    def cleanup_all_processes(self):
        """Comprehensive cleanup of all PX4/Gazebo processes. 
        Call this at the end of optimization or when aborting."""
        print("Performing comprehensive process cleanup...")
        
        processes_to_kill = [
            "px4", "gz", "gzserver", "gzclient", "gazebo",
            "ninja", "make", "cmake", "timeout"
        ]
        
        # Aggressive cleanup - multiple passes
        for attempt in range(5):
            print(f"Cleanup pass {attempt + 1}/5...")
            for process in processes_to_kill:
                os.system(f"pkill -9 {process} >/dev/null 2>&1 || true")
                os.system(f"killall -9 {process} >/dev/null 2>&1 || true")
            
            # Kill specific patterns
            os.system("pkill -9 -f 'PX4-Autopilot' >/dev/null 2>&1 || true")
            os.system("pkill -9 -f 'px4_sitl' >/dev/null 2>&1 || true") 
            os.system("pkill -9 -f 'gz_x500' >/dev/null 2>&1 || true")
            os.system("pkill -9 -f 'cmake.*px4' >/dev/null 2>&1 || true")
            os.system("pkill -9 -f 'make.*px4' >/dev/null 2>&1 || true")
            
            # Kill port users
            os.system("lsof -ti:14540 | xargs kill -9 >/dev/null 2>&1 || true")
            os.system("lsof -ti:14557 | xargs kill -9 >/dev/null 2>&1 || true")
            
            time.sleep(0.5)
        
        print("Comprehensive cleanup complete.")
        time.sleep(2)
        
        # Final verification
        self.check_running_processes()

    def check_running_processes(self):
        """Check what PX4/Gazebo processes are currently running."""
        print("Checking currently running processes...")
        
        # Check for PX4 processes
        px4_result = subprocess.run(["pgrep", "-l", "px4"], capture_output=True, text=True)
        if px4_result.returncode == 0:
            print("Running PX4 processes:")
            for line in px4_result.stdout.strip().split('\n'):
                if line:
                    print(f"  {line}")
        else:
            print("No PX4 processes found")
        
        # Check for Gazebo processes  
        gz_result = subprocess.run(["pgrep", "-l", "gz"], capture_output=True, text=True)
        if gz_result.returncode == 0:
            print("Running Gazebo processes:")
            for line in gz_result.stdout.strip().split('\n'):
                if line:
                    print(f"  {line}")
        else:
            print("No Gazebo processes found")
            
        # Check for build processes
        build_result = subprocess.run(["pgrep", "-lf", "make.*px4"], capture_output=True, text=True)
        if build_result.returncode == 0:
            print("Running build processes:")
            for line in build_result.stdout.strip().split('\n'):
                if line:
                    print(f"  {line}")
        else:
            print("No build processes found")

    def wait_for_px4_sitl_ready(self, timeout=90):
        print("Waiting for PX4 SITL to be ready...")
        start = time.time()
        ready = False
        
        while time.time() - start < timeout:
            # Check for log file or port open
            try:
                with open("/tmp/px4_sitl_gz.log", "r") as f:
                    log = f.read()
                    # Look for multiple indicators of readiness
                    if (("INFO  [uxrce_dds_client] successfully created rt/fmu/out/" in log)or
                        ("INFO  [uxrce_dds_client] synchronized with time offset" in log)):
                        ready = True
                        break
            except Exception:
                pass
            
            # Also check if PX4 process is actually running
            px4_check = subprocess.run(["pgrep", "px4"], capture_output=True)
            if px4_check.returncode != 0:
                print("Warning: PX4 process not found during startup")
                
            time.sleep(2)
        
        if ready:
            time.sleep(15)  # Give it a bit more time to fully stabilize
            print("PX4 SITL is ready!")
        else:
            print("Warning: PX4 SITL may not be fully ready after timeout!")
            # Show current process status for debugging
            self.check_running_processes()
            time.sleep(10)  # Give it more time anyway

    def reset_drone_position(self, x=0, y=0, z=1.5, roll=0, pitch=0, yaw=0):
        """Restart PX4 and Gazebo instead of trying to reset pose (pose setting fails).
        Always restarts PX4/Gazebo to ensure clean state."""
        print("Restarting PX4 and Gazebo (pose setting methods are unreliable)...")
        self.launch_px4_sitl_gazebo()

    # EKF reset methods removed - restarting PX4 handles estimator reset automatically

    def reset_drone_complete(self, x=0, y=0, z=1.5, roll=0, pitch=0, yaw=0, reset_ekf=True):
        """Complete drone reset by restarting PX4 and Gazebo.
        Pose setting methods are unreliable, so always restart instead.
        
        Args:
            x, y, z: Position coordinates (ignored - restart handles positioning)
            roll, pitch, yaw: Orientation (ignored - restart handles positioning)
            reset_ekf: Whether to also reset PX4's EKF estimator (ignored - restart handles this)
        """
        print("Performing complete drone reset by restarting PX4 and Gazebo...")
        self.launch_px4_sitl_gazebo()
        print("Complete drone reset finished (via restart).")

    # Define full parameter search space as a class attribute
    base_search_space = [
        Real(0.05, 0.8, name='ndi_kp'),      # NDI proportional gain
        Real(0.0, 0.1, name='ndi_kd'),      # NDI derivative gain  
        Real(0.0, 0.2, name='fb_kp'),        # Feedback proportional
        Real(0.0, 0.02, name='fb_kd'),       # Feedback derivative
        Real(0.1, 7.0, name='att_kp'),       # Attitude proportional
        Real(0.0, 5.0, name='att_kd'),       # Attitude derivative
    ]

    # Map parameter names to control_params.json paths (section, key)
    PARAM_TO_CONFIG = {
        'ndi_kp': ('ndi_rate', 'kp'),
        'ndi_kd': ('ndi_rate', 'kd'),
        'fb_kp':  ('feedback_rate', 'kp'),
        'fb_kd':  ('feedback_rate', 'kd'),
        'att_kp': ('attitude', 'kp'),
        'att_kd': ('attitude', 'kd'),
    }

    def __init__(self, config_file: str, flight_data_dir: str,
                 reset_history: Optional[bool] = None,
                 max_seed_cost: Optional[float] = None,
                 results_file: Optional[str] = None,
                 tune_flags: Optional[Dict[str, bool]] = None):
        self.config_file = config_file
        self.flight_data_dir = flight_data_dir
        self.base_config = self.load_config()
        # Performance tracking
        self.trial_results = []
        self.best_params = None
        self.best_cost = float('inf')
        # Graceful shutdown control
        self.stop_requested = False
        self.in_trial = False
        # Results database paths and cache
        self.results_output_dir = os.path.join(self.flight_data_dir, 'outputs')
        os.makedirs(self.results_output_dir, exist_ok=True)
        self.results_file = results_file or os.path.join(self.results_output_dir, 'tuning_results.json')
        # Cache of evaluated params -> cost (from prior DB and current session)
        self.evaluated_cache: Dict[Tuple[float, ...], float] = {}
        # Prior data for seeding the optimizer
        self._seed_x0: List[List[float]] = []
        self._seed_y0: List[float] = []
        # Behavior controls
        self.reset_history = (
            reset_history
            if reset_history is not None
            else os.environ.get('AUTOTUNE_RESET_HISTORY', '').lower() in ('1', 'true', 'yes')
        )
        self.max_seed_cost = (
            float(max_seed_cost)
            if max_seed_cost is not None
            else float(os.environ.get('AUTOTUNE_MAX_SEED_COST', '60'))
        )
        # Tune flags and active search space
        self.tune_flags = self._resolve_tune_flags(tune_flags)
        self.active_space = [d for d in self.base_search_space if self.tune_flags.get(d.name, True)]
        if not self.active_space:
            raise ValueError("No parameters selected for tuning. Enable at least one in tune_flags.json or via CLI.")
        # Load any existing database to seed optimization
        self._load_previous_results()

        # If we have a historical best, apply it to the active config (warm start)
        if self.best_params is not None:
            try:
                self.update_config(self.best_params)
                print(f"Warm-starting from historical best (cost={self.best_cost:.4f}). Applied to config.")
            except Exception as e:
                print(f"Warning: Could not apply historical best params to config: {e}")
        else:
            # Ensure config contains current values (no-op, but validates mapping)
            try:
                self.update_config([])  # Update with nothing to validate
            except Exception:
                pass

    def _resolve_tune_flags(self, provided: Optional[Dict[str, bool]]) -> Dict[str, bool]:
        """Resolve tune flags priority: provided > file > default(all True)."""
        names = [d.name for d in self.base_search_space]
        if provided:
            # Fill missing with True by default
            return {n: bool(provided.get(n, True)) for n in names}
        # Default tune_flags.json path
        default_flags_path = os.path.join(self.flight_data_dir, 'config', 'tune_flags.json')
        flags = None
        try:
            if os.path.isfile(default_flags_path):
                with open(default_flags_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    flags = {n: bool(data.get(n, True)) for n in names}
        except Exception as e:
            print(f"Warning: Could not read tune flags file: {e}")
        if flags is None:
            # Default to all True and write a template for the user
            flags = {n: True for n in names}
            try:
                os.makedirs(os.path.dirname(default_flags_path), exist_ok=True)
                with open(default_flags_path, 'w') as f:
                    json.dump(flags, f, indent=2)
                print(f"Wrote default tune flags to {default_flags_path}")
            except Exception as e:
                print(f"Warning: Could not write default tune_flags.json: {e}")
        print("Active tune flags:", flags)
        return flags
        
    def load_config(self) -> Dict:
        """Load current configuration"""
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def update_config(self, params: List[float]) -> None:
        """Update configuration with new parameters.
        Applies only active parameters; inactive ones remain unchanged.
        """
        # Start from current file on disk to preserve updates between trials
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
        except Exception:
            config = self.base_config.copy()

        # Build dict of current param values (for debug) and apply updates
        names = self._names()
        if params and len(params) != len(names):
            raise ValueError(f"Expected {len(names)} parameters, got {len(params)}")
        updates = dict(zip(names, params)) if params else {}

        for pname, (section, key) in self.PARAM_TO_CONFIG.items():
            if pname in updates:
                # Ensure section exists
                if section not in config:
                    config[section] = {}
                try:
                    config[section][key] = float(updates[pname])
                except Exception:
                    pass  # Skip invalid value silently

        # Write updated config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def _load_previous_results(self) -> None:
        """Load previous tuning results to seed optimizer and initialize cache/best."""
        try:
            if self.reset_history or not os.path.isfile(self.results_file):
                print("History disabled or no results DB found; starting fresh.")
                return
            with open(self.results_file, 'r') as f:
                db = json.load(f)
            trials = db.get('trial_history', []) or []
            names = self._names()
            # Collect cache of all valid trials; seed x0/y0 only from low-cost subset
            for t in trials:
                if t is None or t.get('failed'):
                    continue
                params_dict = t.get('params')
                cost = t.get('cost')
                if params_dict is None or cost is None:
                    continue
                # Ensure all names exist in params_dict
                if not all(n in params_dict for n in names):
                    continue
                vec = self._dict_to_vector(params_dict)
                key = self._canon_key(vec)
                # Always update cache to avoid recomputing known points
                try:
                    c = float(cost)
                except Exception:
                    continue
                prev_cost = self.evaluated_cache.get(key)
                if prev_cost is None or c < prev_cost:
                    self.evaluated_cache[key] = c
                # Seed optimizer only if within acceptable cost threshold
                if c <= self.max_seed_cost:
                    self._seed_x0.append(vec)
                    self._seed_y0.append(c)

            # Initialize current best from DB best or from min of trials
            db_best_cost = db.get('best_cost')
            db_best_params = db.get('best_params')
            if db_best_cost is not None and db_best_params is not None and all(n in db_best_params for n in names):
                self.best_cost = float(db_best_cost)
                self.best_params = self._dict_to_vector(db_best_params)
            else:
                # Fallback to best from trials
                if self._seed_y0:
                    min_idx = int(np.argmin(self._seed_y0))
                    self.best_cost = float(self._seed_y0[min_idx])
                    self.best_params = list(self._seed_x0[min_idx])

            print(f"History loaded: {len(self._seed_x0)} seed points from {len(trials)} prior trials. Cache size: {len(self.evaluated_cache)}")
        except Exception as e:
            print(f"Warning: Could not load previous results DB: {e}")
    
    def run_flight_test(self, duration: float = 22.0, reset_sim=True) -> str:
        """Run a flight test and return log filename.
        - No rebuild (assumes workspace was built beforehand).
        - Restart only PX4 SITL + controller node per trial.
        """
        print(f"Starting flight test with duration {duration}s...")

        # 1) Kill previous instances (PX4, Gazebo, controller)
        kill_patterns = [
            "px4", "gz", "gzserver", "gzclient", "gazebo",
            "offboard_control_spin_tt"
        ]
        for pat in kill_patterns:
            subprocess.run(f"pkill -f {pat}", shell=True, stderr=subprocess.DEVNULL)
        time.sleep(2)

        # 2) Launch PX4 SITL + Gazebo (gz_x500)
        print("Launching PX4 SITL + Gazebo...")
        px4 = subprocess.Popen(
            "HEADLESS=1 make px4_sitl gz_x500 > /tmp/px4_sitl_gz.log 2>&1",
            cwd="/home/pyro/PX4-Autopilot",
            shell=True,
            executable="/bin/bash",
        )

        # Wait for PX4 readiness (reads /tmp/px4_sitl_gz.log)
        self.wait_for_px4_sitl_ready(timeout=60)

        # 3) Launch controller (already built)
        print("Launching controller node...")
        ctrl = subprocess.Popen(
            "source install/local_setup.bash && ros2 run px4_ros_com offboard_control_spin_tt",
            cwd="/home/pyro/ws_offboard_control",
            shell=True,
            executable="/bin/bash",
        )

        # 4) Let it fly; allow enough time for controller to save logs internally
        # Controller saves logs and shuts down around 20s after arming; we use a cushion
        start_time = time.time()
        time.sleep(max(0.0, duration))

        # 5) Prefer controller self-shutdown; if still alive, nudge with SIGINT then SIGTERM
        print("Waiting for controller to finish and save logs...")
        wait_deadline = time.time() + 15.0
        while time.time() < wait_deadline:
            if ctrl.poll() is not None:
                break
            time.sleep(0.5)
        if ctrl.poll() is None:
            print("Controller still running, sending SIGINT...")
            try:
                ctrl.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                ctrl.wait(timeout=8)
            except Exception:
                pass
        if ctrl.poll() is None:
            print("Controller still running, sending SIGTERM...")
            try:
                ctrl.terminate()
                ctrl.wait(timeout=5)
            except Exception:
                ctrl.kill()

        # Give PX4 a moment and then terminate
        try:
            px4.terminate()
            px4.wait(timeout=5)
        except Exception:
            px4.kill()

        # 6) Wait for a new log file created after start_time
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
                    # pick latest modified
                    log_path = max(candidates, key=os.path.getmtime)
                    # ensure file has content
                    if os.path.getsize(log_path) < 100:
                        log_path = None
                if log_path is None:
                    time.sleep(0.5)
            except FileNotFoundError:
                time.sleep(0.5)

        if log_path is None:
            raise RuntimeError("No log file generated (timeout waiting for CSV)")

        # Truncate PX4 SITL log so next trial starts with a clean log
        try:
            with open("/tmp/px4_sitl_gz.log", "w"):
                pass  # Truncate file
        except Exception as e:
            print(f"Warning: Could not truncate PX4 SITL log: {e}")

        return log_path

    
    def analyze_performance(self, log_file: str) -> float:

        try:
            import pandas as pd
            df = pd.read_csv(log_file)

            if len(df) < 10:
                return 1000.0

            # Extract by column name
            time = df['timestamp'].to_numpy()
            pos_x_err = df['pos_err_x'].to_numpy()
            pos_y_err = df['pos_err_y'].to_numpy()
            pos_z_err = df['pos_err_z'].to_numpy()
            roll_err = df['att_err_roll'].to_numpy()
            pitch_err = df['att_err_pitch'].to_numpy()
            p_rate_err = df['rate_err_p'].to_numpy()
            q_rate_err = df['rate_err_q'].to_numpy()
            r_rate_err = df['rate_err_r'].to_numpy()
            thrust_x = df['thrust_x'].to_numpy()
            thrust_y = df['thrust_y'].to_numpy()
            thrust_z = df['thrust_z'].to_numpy()
            torque_x = df['torque_x'].to_numpy()
            torque_y = df['torque_y'].to_numpy()
            torque_z = df['torque_z'].to_numpy()
            pos_z = df['pos_z'].to_numpy()
            vel_z = df['vel_z'].to_numpy() if 'vel_z' in df.columns else np.zeros_like(pos_z)

            # Compute cost function components
            pos_z_s = np.sqrt(np.mean(pos_z_err**2))
            att_rms = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
            rate_rms = np.sqrt(np.mean(p_rate_err**2 + q_rate_err**2 + r_rate_err**2))

            # 2. Control Smoothness (derivatives)
            thrust_smoothness = np.mean(np.diff(thrust_x)**2 + np.diff(thrust_y)**2 + np.diff(thrust_z)**2)
            torque_smoothness = np.mean(np.diff(torque_x)**2 + np.diff(torque_y)**2 + np.diff(torque_z)**2)

            # 3. Control Effort
            control_effort = np.mean(thrust_x**2 + thrust_y**2 + thrust_z**2 +
                                     torque_x**2 + torque_y**2 + torque_z**2)

            # 4. Stability (variance in late part of flight)
            if len(df) > 100:
                stable_period = df.iloc[-50:, :]
                stability_cost = np.var(stable_period[['pos_err_x', 'pos_err_y', 'pos_err_z']].to_numpy())
            else:
                stability_cost = 10.0

            # Weighted cost function (base)
            cost = (
                18.0 * pos_z_s +          # Position tracking
                1.0 * att_rms +           # Attitude tracking
                0.2 * rate_rms +          # Rate tracking
                0.1 * thrust_smoothness + # Control smoothness
                0.2 * torque_smoothness +
                0.05 * control_effort +   # Control effort
                0.5 * stability_cost      # Stability
            )


            return float(cost)

        except Exception as e:
            print(f"Error analyzing performance: {e}")
            return 1000.0  # High cost for failed analysis
    
    def calculate_oscillation_penalty(self, signal_x: np.ndarray, signal_y: np.ndarray, dt: float) -> float:
        """Calculate penalty for high-frequency oscillations"""
        try:
            # Simple high-pass filter to detect oscillations
            if len(signal_x) < 20:
                return 0.0
                
            # Calculate second derivatives (acceleration of torque commands)
            acc_x = np.diff(signal_x, 2) / (dt**2)
            acc_y = np.diff(signal_y, 2) / (dt**2)
            
            # Penalty based on RMS of second derivatives
            oscillation_penalty = 0.1 * np.sqrt(np.mean(acc_x**2 + acc_y**2))
            
            return min(oscillation_penalty, 50.0)  # Cap the penalty
            
        except:
            return 0.0
    
    def objective_function(self, param_values) -> float:
        """Objective function for Bayesian optimization (param_values: list of floats)"""
        # If a stop was requested before starting a new trial, abort cleanly
        if self.stop_requested:
            print("Stop requested before starting new trial. Aborting optimization...", flush=True)
            raise KeyboardInterrupt

        trial_idx = len(self.trial_results) + 1
        print(f"\nTrial {trial_idx}")
        print(f"Testing parameters: {dict(zip([d.name for d in self.active_space], param_values))}")
        sys.stdout.flush()
        # Skip evaluation if we've already seen these parameters
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
            # Update best if applicable
            if cached_cost < self.best_cost:
                self.best_cost = cached_cost
                # Store as list of native floats
                self.best_params = [float(v) for v in param_values]
                print(f"New best cost (from cache): {cached_cost:.4f}")
            return cached_cost
        # Update configuration
        self.update_config(param_values)
        # Run flight test
        self.in_trial = True
        try:
            log_file = self.run_flight_test(duration=20.0)
            cost = self.analyze_performance(log_file)
            # Derive expected PDF analysis path from log file name
            base_name = os.path.splitext(os.path.basename(log_file))[0]
            pdf_path = os.path.join(self.results_output_dir, f"{base_name}_analysis.pdf")
            # Track results
            result = {
                'params': self._vector_to_dict(param_values),
                'cost': cost,
                # Link to the PDF analysis report instead of the raw CSV log
                'log_file': pdf_path
            }
            self.trial_results.append(result)
            # Update cache
            self.evaluated_cache[key] = float(cost)
            # Update best parameters
            if cost < self.best_cost:
                self.best_cost = cost
                # Store as list of native floats
                self.best_params = [float(v) for v in param_values]
                print(f"New best cost: {cost:.4f}")
            print(f"Cost: {cost:.4f}")
            sys.stdout.flush()
            # If a stop was requested during this trial, exit now that cost is printed
            if self.stop_requested:
                print("Stop flag detected. Exiting after printing cost.", flush=True)
                raise KeyboardInterrupt
            return cost
        except Exception as e:
            # Even on failure, emit a cost so the user sees the outcome
            cost = 1000.0
            print(f"Trial failed: {e}")
            print(f"Cost: {cost:.4f} (failure)")
            sys.stdout.flush()
            # Record failed trial too
            self.trial_results.append({
                'params': self._vector_to_dict(param_values),
                'cost': cost,
                'log_file': None,
                'failed': True,
                'error': str(e)
            })
            # Update cache with failure cost too to avoid immediate re-tries
            self.evaluated_cache[key] = float(cost)
            # If stop requested, break after reporting
            if self.stop_requested:
                raise KeyboardInterrupt
            return cost  # High cost for failed trials
        finally:
            self.in_trial = False
    
    def optimize(self, n_calls: int = 50) -> Dict:
        """Run Bayesian optimization"""
        print(f"Starting Bayesian optimization with {n_calls} trials...")
        
        try:
            # Prepare prior evaluations to seed the optimizer (deduplicated)
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

            # Avoid wasting random initial points if we already have seeds
            n_init_default = 10
            n_init = max(0, n_init_default - len(x0)) if x0 else n_init_default

            result = gp_minimize(
                func=self.objective_function,
                dimensions=self.active_space,
                x0=x0 if x0 else None,
                y0=y0 if x0 else None,
                n_calls=n_calls,
                n_initial_points=n_init,
                acq_func='gp_hedge',
                acq_optimizer='sampling',
                random_state=42
            )
            
            # Update config with best parameters
            if self.best_params is not None:
                self.update_config(self.best_params)
                print(f"\nOptimization complete!")
                print(f"Best cost: {self.best_cost:.4f}")
                print(f"Best parameters: {dict(zip([d.name for d in self.active_space], self.best_params))}")
                
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
            # Always cleanup processes when optimization ends
            print("\nPerforming final cleanup...")
            self.cleanup_all_processes()

    def save_results(self, results: Dict, filename: str) -> None:
        """Merge-save optimization results into a persistent JSON database.
        - Deduplicate trials by canonicalized parameter vector, keeping the lowest cost.
        - Recompute best from all trials when saving.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        names = self._names()

        # Load existing DB if present
        existing = {}
        if os.path.isfile(filename):
            try:
                with open(filename, 'r') as f:
                    existing = json.load(f) or {}
            except Exception as e:
                print(f"Warning: Could not read existing results file: {e}")
                existing = {}

        existing_trials = existing.get('trial_history', []) or []
        new_trials = results.get('all_results', []) or []
        combined: Dict[Tuple[float, ...], Dict] = {}

        def to_key_from_trial(tr: Dict) -> Tuple[float, ...]:
            pd = tr.get('params') or {}
            if not all(n in pd for n in names):
                # Incomplete params; use empty to avoid crash, but won't dedupe
                return tuple()
            vec = [float(pd[n]) for n in names]
            return self._canon_key(vec)

        # Insert existing trials
        for tr in existing_trials:
            key = to_key_from_trial(tr)
            if key == tuple():
                continue
            cost = tr.get('cost')
            # Keep first for now; we'll resolve with new ones below
            if key not in combined:
                combined[key] = tr
            else:
                # Keep lower cost
                try:
                    if cost is not None and combined[key].get('cost') is not None and float(cost) < float(combined[key]['cost']):
                        combined[key] = tr
                except Exception:
                    pass

        # Merge new trials, prefer lower cost
        for tr in new_trials:
            key = to_key_from_trial(tr)
            if key == tuple():
                continue
            cost = tr.get('cost')
            if key not in combined:
                combined[key] = tr
            else:
                try:
                    if cost is not None and combined[key].get('cost') is not None and float(cost) < float(combined[key]['cost']):
                        combined[key] = tr
                except Exception:
                    pass

        # Build merged list and recompute best
        merged_trials = list(combined.values())
        # Recompute best across all successful trials
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
                dict(zip(names, [float(x) for x in results.get('best_params')])) if results.get('best_params') else existing.get('best_params')
            ),
            'trial_history': merged_trials
        }

        try:
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results to {filename}: {e}")

def main():
    """Run automated tuning"""
    parser = argparse.ArgumentParser(description="Automated Controller Tuning using Bayesian Optimization")
    parser.add_argument("--use-history", dest="use_history", action="store_true", help="Use prior tuning_results.json to warm start", default=True)
    parser.add_argument("--no-history", dest="use_history", action="store_false", help="Ignore history and start fresh")
    parser.add_argument("--max-seed-cost", type=float, default=None, help="Max cost from history to seed the optimizer")
    parser.add_argument("--results-file", type=str, default=None, help="Path to results DB JSON (defaults to flight_data/outputs/tuning_results.json)")
    parser.add_argument("--trials", type=int, default=60, help="Number of optimization calls")
    parser.add_argument("--tune-flags", type=str, default=None, help="Path to tune_flags.json; overrides defaults if provided")
    parser.add_argument("--tune-only", type=str, default=None, help="Comma-separated list of parameter names to tune (others disabled)")
    parser.add_argument("--tune-except", type=str, default=None, help="Comma-separated list of parameter names to disable (others enabled)")
    parser.add_argument("--list-params", action="store_true", help="List available parameter names and exit")
    args = parser.parse_args()

    config_file = "/home/pyro/ws_offboard_control/flight_data/config/control_params.json"
    flight_data_dir = "/home/pyro/ws_offboard_control/flight_data"

    # Generate tune flags based on CLI, if any
    cli_flags: Optional[Dict[str, bool]] = None
    names = [d.name for d in ControllerAutoTuner.base_search_space]
    if args.list_params:
        print("Available parameters:")
        for n in names:
            print(f" - {n}")
        return
    if args.tune_only:
        only = [s.strip() for s in args.tune_only.split(',') if s.strip()]
        invalid = [x for x in only if x not in names]
        if invalid:
            raise SystemExit(f"Invalid names in --tune-only: {invalid}")
        cli_flags = {n: (n in only) for n in names}
    elif args.tune_except:
        exc = [s.strip() for s in args.tune_except.split(',') if s.strip()]
        invalid = [x for x in exc if x not in names]
        if invalid:
            raise SystemExit(f"Invalid names in --tune-except: {invalid}")
        cli_flags = {n: (n not in exc) for n in names}
    elif args.tune_flags:
        try:
            with open(args.tune_flags, 'r') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("tune_flags file must contain a JSON object of {name: bool}")
            cli_flags = {n: bool(data.get(n, True)) for n in names}
        except Exception as e:
            raise SystemExit(f"Failed to read --tune-flags file: {e}")
    
    tuner = ControllerAutoTuner(
        config_file,
        flight_data_dir,
        reset_history=(not args.use_history),
        max_seed_cost=args.max_seed_cost,
        results_file=args.results_file,
        tune_flags=cli_flags
    )
    # Ensure we survive until each trial prints the cost
    tuner.install_signal_handlers()
    
    # Run optimization
    results = tuner.optimize(n_calls=args.trials)
    
    # Save results
    out_file = args.results_file or f"{flight_data_dir}/outputs/tuning_results.json"
    tuner.save_results(results, out_file)

if __name__ == "__main__":
    main()