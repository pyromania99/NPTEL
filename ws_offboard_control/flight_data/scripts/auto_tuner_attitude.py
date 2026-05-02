#!/usr/bin/env python3
"""
Automated Attitude Controller & Gain Scheduling Tuning
Focuses on optimizing:
1. Attitude base gains (kp, kd)
2. Gain scheduling parameters (kp_min, kd_max, rate_threshold)
"""

import numpy as np
import matplotlib.pyplot as plt
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

class AttitudeAutoTuner:
    """Auto-tuner specialized for attitude controller and gain scheduling"""
    
    # Define parameter search space - ONLY optimizing gain scheduling parameters
    search_space = [
        Real(0.3, 1.5, name='gs_kp_min'),        # Gain scheduling: min P (at high rate)
        Real(3.5, 6.0, name='gs_kd_max'),        # Gain scheduling: max D (at high rate)
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
                 results_file: Optional[str] = None):
        self.config_file = config_file
        self.flight_data_dir = flight_data_dir
        self.base_config = self.load_config()
        
        # Performance tracking
        self.trial_results = []
        self.best_params = None
        self.best_cost = float('inf')
        
        # Graceful shutdown
        self.stop_requested = False
        self.in_trial = False
        self._signals_received = 0
        
        # Results database
        self.results_output_dir = os.path.join(self.flight_data_dir, 'outputs')
        os.makedirs(self.results_output_dir, exist_ok=True)
        self.results_file = results_file or os.path.join(
            self.results_output_dir, 'tuning_results_attitude.json'
        )
        
        # Cache and seeds
        self.evaluated_cache: Dict[Tuple[float, ...], float] = {}
        self._seed_x0: List[List[float]] = []
        self._seed_y0: List[float] = []
        
        # Behavior controls
        self.reset_history = (
            reset_history if reset_history is not None
            else os.environ.get('AUTOTUNE_RESET_HISTORY', '').lower() in ('1', 'true', 'yes')
        )
        self.max_seed_cost = (
            float(max_seed_cost) if max_seed_cost is not None
            else float(os.environ.get('AUTOTUNE_MAX_SEED_COST', '50'))
        )
        
        # Load history
        self._load_previous_results()
        
        # Warm start from historical best
        if self.best_params is not None:
            try:
                self.update_config(self.best_params)
                print(f"Warm-starting from historical best (cost={self.best_cost:.4f})")
            except Exception as e:
                print(f"Warning: Could not apply historical best: {e}")
    
    def install_signal_handlers(self):
        """Install handlers for graceful shutdown"""
        def _handler(sig, frame):
            self._signals_received += 1
            name = signal.Signals(sig).name
            if self._signals_received == 1:
                self.stop_requested = True
                print(f"\nReceived {name}. Will stop after current trial...", flush=True)
            else:
                print(f"\nReceived {name} again. Forcing exit...", flush=True)
                raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    
    def load_config(self) -> Dict:
        """Load current configuration"""
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def update_config(self, params: List[float]) -> None:
        """Update configuration with new parameters
        
        params order:
        [0] gs_kp_min        - Min P at high rate
        [1] gs_kd_max        - Max D at high rate
        
        NOTE: Base gains (kp_base, kd_base) and rate_threshold are kept fixed
        """
        config = self.base_config.copy()
        
        # Keep base gains unchanged from config file
        att_kp_base = config['attitude']['kp']
        att_kd_base = config['attitude']['kd']
        
        # Update gain scheduling parameters only
        if 'gain_scheduling' not in config:
            config['gain_scheduling'] = {'enabled': True}
        
        config['gain_scheduling']['enabled'] = True
        config['gain_scheduling']['kp_min'] = params[0]
        config['gain_scheduling']['kd_max'] = params[1]
        # Keep rate_threshold from existing config or use default
        if 'rate_threshold' not in config['gain_scheduling']:
            config['gain_scheduling']['rate_threshold'] = 5.0
        
        rate_threshold = config['gain_scheduling']['rate_threshold']
        
        # Write updated config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Updated config: gs_kp_min={params[0]:.3f}, gs_kd_max={params[1]:.3f} "
              f"(keeping att_kp={att_kp_base:.3f}, att_kd={att_kd_base:.3f}, "
              f"rate_thresh={rate_threshold:.3f})")
    
    def _load_previous_results(self) -> None:
        """Load previous tuning results"""
        try:
            if self.reset_history or not os.path.isfile(self.results_file):
                print("Starting fresh (no history)")
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
                except:
                    continue
                
                # Update cache
                prev = self.evaluated_cache.get(key)
                if prev is None or c < prev:
                    self.evaluated_cache[key] = c
                
                # Seed optimizer if cost is good
                if c <= self.max_seed_cost:
                    self._seed_x0.append(vec)
                    self._seed_y0.append(c)
            
            # Initialize best from DB
            db_best_cost = db.get('best_cost')
            db_best_params = db.get('best_params')
            
            if (db_best_cost is not None and db_best_params is not None 
                and all(n in db_best_params for n in names)):
                self.best_cost = float(db_best_cost)
                self.best_params = self._dict_to_vector(db_best_params)
            elif self._seed_y0:
                min_idx = int(np.argmin(self._seed_y0))
                self.best_cost = float(self._seed_y0[min_idx])
                self.best_params = list(self._seed_x0[min_idx])
            
            print(f"Loaded {len(self._seed_x0)} seed points from {len(trials)} trials. "
                  f"Cache: {len(self.evaluated_cache)}")
        
        except Exception as e:
            print(f"Warning: Could not load history: {e}")
    
    def cleanup_processes(self):
        """Kill PX4, Gazebo, and controller processes"""
        print("Cleaning up processes...")
        
        patterns = ["px4", "gz", "gzserver", "gzclient", "gazebo", 
                   "offboard_control_spin_tt"]
        
        for pattern in patterns:
            subprocess.run(f"pkill -9 {pattern}", shell=True, 
                         stderr=subprocess.DEVNULL)
        
        time.sleep(2)
    
    def wait_for_px4_ready(self, timeout=60):
        """Wait for PX4 SITL to be ready"""
        print("Waiting for PX4 SITL...")
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                with open("/tmp/px4_sitl_gz.log", "r") as f:
                    log = f.read()
                    if "INFO  [uxrce_dds_client] successfully created" in log:
                        print("PX4 SITL ready!")
                        time.sleep(5)
                        return True
            except:
                pass
            time.sleep(1)
        
        print("Warning: PX4 may not be fully ready")
        return False
    
    def run_flight_test(self, duration: float = 22.0) -> str:
        """Run flight test and return log filename"""
        print(f"Starting flight test ({duration}s)...")
        
        # 1. Cleanup
        self.cleanup_processes()
        
        # 2. Launch PX4 + Gazebo
        print("Launching PX4 SITL + Gazebo...")
        px4 = subprocess.Popen(
            "HEADLESS=1 make px4_sitl gz_x500 > /tmp/px4_sitl_gz.log 2>&1",
            cwd="/home/pyro/PX4-Autopilot",
            shell=True,
            executable="/bin/bash"
        )
        
        self.wait_for_px4_ready()
        
        # 3. Launch controller
        print("Launching controller...")
        ctrl = subprocess.Popen(
            "source install/local_setup.bash && "
            "ros2 run px4_ros_com offboard_control_spin_tt",
            cwd="/home/pyro/ws_offboard_control",
            shell=True,
            executable="/bin/bash"
        )
        
        # 4. Let it fly
        start_time = time.time()
        time.sleep(duration)
        
        # 5. Stop controller
        print("Stopping controller...")
        wait_deadline = time.time() + 10
        while time.time() < wait_deadline:
            if ctrl.poll() is not None:
                break
            time.sleep(0.5)
        
        if ctrl.poll() is None:
            ctrl.send_signal(signal.SIGINT)
            try:
                ctrl.wait(timeout=5)
            except:
                ctrl.kill()
        
        # 6. Stop PX4
        try:
            px4.terminate()
            px4.wait(timeout=5)
        except:
            px4.kill()
        
        # 7. Find log file
        log_dir = f"{self.flight_data_dir}/logs"
        print("Looking for log file...")
        
        log_path = None
        deadline = time.time() + 30
        
        while time.time() < deadline and log_path is None:
            try:
                candidates = []
                for f in os.listdir(log_dir):
                    if f.endswith('.csv'):
                        full = os.path.join(log_dir, f)
                        if os.path.getmtime(full) >= start_time:
                            candidates.append(full)
                
                if candidates:
                    log_path = max(candidates, key=os.path.getmtime)
                    if os.path.getsize(log_path) < 100:
                        log_path = None
            except:
                pass
            
            if log_path is None:
                time.sleep(0.5)
        
        if log_path is None:
            raise RuntimeError("No log file generated")
        
        print(f"Log file: {log_path}")
        return log_path
    
    def analyze_performance(self, log_file: str) -> float:
        """Analyze flight performance - FOCUSED ON ATTITUDE TRACKING"""
        try:
            import pandas as pd
            df = pd.read_csv(log_file)
            
            if len(df) < 10:
                return 1000.0
            
            # Extract data
            time = df['timestamp'].to_numpy()
            roll_err = df['att_err_roll'].to_numpy()
            pitch_err = df['att_err_pitch'].to_numpy()
            p_rate_err = df['rate_err_p'].to_numpy()
            q_rate_err = df['rate_err_q'].to_numpy()
            r_rate_err = df['rate_err_r'].to_numpy()
            torque_x = df['torque_x'].to_numpy()
            torque_y = df['torque_y'].to_numpy()
            torque_z = df['torque_z'].to_numpy()
            
            # Get yaw rate for gain scheduling analysis
            r_cmd = df['rate_sp_r'].to_numpy() if 'rate_sp_r' in df.columns else np.zeros_like(time)
            
            # === PRIMARY METRICS: ATTITUDE TRACKING ===
            # Roll/pitch tracking (most important)
            att_rms = np.sqrt(np.mean(roll_err**2 + pitch_err**2))
            
            # Rate tracking
            rate_rms = np.sqrt(np.mean(p_rate_err**2 + q_rate_err**2))
            
            # Yaw rate tracking (for gain scheduling effectiveness)
            yaw_rate_rms = np.sqrt(np.mean(r_rate_err**2))
            
            # === SECONDARY METRICS: CONTROL QUALITY ===
            # Torque smoothness (penalize oscillations)
            torque_jerk = np.mean(np.diff(torque_x)**2 + np.diff(torque_y)**2)
            
            # Control effort
            control_effort = np.mean(torque_x**2 + torque_y**2 + torque_z**2)
            
            # === GAIN SCHEDULING PERFORMANCE ===
            # Separate high-rate and low-rate regions
            high_rate_mask = np.abs(r_cmd) > 3.0
            low_rate_mask = np.abs(r_cmd) < 1.0
            
            high_rate_att_err = 0.0
            low_rate_att_err = 0.0
            
            if np.any(high_rate_mask):
                high_rate_att_err = np.sqrt(np.mean(
                    roll_err[high_rate_mask]**2 + pitch_err[high_rate_mask]**2
                ))
            
            if np.any(low_rate_mask):
                low_rate_att_err = np.sqrt(np.mean(
                    roll_err[low_rate_mask]**2 + pitch_err[low_rate_mask]**2
                ))
            
            # === COST FUNCTION (ATTITUDE FOCUSED) ===
            cost = (
                30.0 * att_rms +              # PRIMARY: Roll/pitch tracking
                5.0 * rate_rms +               # Rate tracking (roll/pitch)
                3.0 * yaw_rate_rms +           # Yaw rate tracking
                0.5 * torque_jerk +            # Smoothness
                0.1 * control_effort +         # Efficiency
                10.0 * high_rate_att_err +     # High-rate performance
                5.0 * low_rate_att_err         # Low-rate performance
            )
            
            print(f"  Attitude RMS: {att_rms:.4f} rad")
            print(f"  Rate RMS: {rate_rms:.4f} rad/s")
            print(f"  Yaw rate RMS: {yaw_rate_rms:.4f} rad/s")
            print(f"  High-rate att err: {high_rate_att_err:.4f} rad")
            print(f"  Low-rate att err: {low_rate_att_err:.4f} rad")
            
            return float(cost)
        
        except Exception as e:
            print(f"Error analyzing performance: {e}")
            import traceback
            traceback.print_exc()
            return 1000.0
    
    def objective_function(self, param_values) -> float:
        """Objective function for optimization"""
        if self.stop_requested:
            print("Stop requested, aborting...")
            raise KeyboardInterrupt
        
        trial_idx = len(self.trial_results) + 1
        print(f"\n{'='*70}")
        print(f"TRIAL {trial_idx}")
        print(f"{'='*70}")
        
        params_dict = self._vector_to_dict(param_values)
        print("Parameters:")
        for name, val in params_dict.items():
            print(f"  {name:20s}: {val:.4f}")
        
        sys.stdout.flush()
        
        # Check cache
        key = self._canon_key(param_values)
        if key in self.evaluated_cache:
            cost = self.evaluated_cache[key]
            print(f"Using cached cost: {cost:.4f}")
            
            result = {
                'params': params_dict,
                'cost': cost,
                'log_file': None,
                'cached': True
            }
            self.trial_results.append(result)
            
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_params = list(param_values)
                print(f"New best (cached): {cost:.4f}")
            
            return cost
        
        # Run trial
        self.in_trial = True
        try:
            self.update_config(param_values)
            log_file = self.run_flight_test(duration=22.0)
            cost = self.analyze_performance(log_file)
            
            # Track results
            result = {
                'params': params_dict,
                'cost': cost,
                'log_file': log_file
            }
            self.trial_results.append(result)
            self.evaluated_cache[key] = float(cost)
            
            # Update best
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_params = list(param_values)
                print(f"\n*** NEW BEST COST: {cost:.4f} ***\n")
            
            print(f"Cost: {cost:.4f}")
            sys.stdout.flush()
            
            if self.stop_requested:
                raise KeyboardInterrupt
            
            return cost
        
        except Exception as e:
            cost = 1000.0
            print(f"Trial failed: {e}")
            print(f"Cost: {cost:.4f} (failure)")
            
            self.trial_results.append({
                'params': params_dict,
                'cost': cost,
                'log_file': None,
                'failed': True,
                'error': str(e)
            })
            self.evaluated_cache[key] = float(cost)
            
            if self.stop_requested:
                raise KeyboardInterrupt
            
            return cost
        
        finally:
            self.in_trial = False
    
    def optimize(self, n_calls: int = 30) -> Dict:
        """Run Bayesian optimization"""
        print(f"\n{'='*70}")
        print(f"ATTITUDE GAIN SCHEDULING AUTO-TUNER")
        print(f"{'='*70}")
        print(f"Optimizing: gs_kp_min, gs_kd_max (2 parameters)")
        print(f"Fixed: att_kp_base, att_kd_base, rate_threshold")
        print(f"Trials: {n_calls}")
        print(f"{'='*70}\n")
        
        try:
            # Prepare seeds
            x0, y0 = [], []
            seen = {}
            
            for x, y in zip(self._seed_x0, self._seed_y0):
                k = self._canon_key(x)
                if k not in seen or y < seen[k]:
                    seen[k] = float(y)
            
            if seen:
                for k, y in seen.items():
                    x0.append(list(k))
                    y0.append(float(y))
            
            n_init = max(0, 10 - len(x0)) if x0 else 10
            
            print(f"Starting optimization with {len(x0)} seed points\n")
            
            result = gp_minimize(
                func=self.objective_function,
                dimensions=self.search_space,
                x0=x0 if x0 else None,
                y0=y0 if x0 else None,
                n_calls=n_calls,
                n_initial_points=n_init,
                acq_func='gp_hedge',
                acq_optimizer='sampling',
                random_state=42
            )
            
            # Apply best
            if self.best_params is not None:
                self.update_config(self.best_params)
                print(f"\n{'='*70}")
                print(f"OPTIMIZATION COMPLETE")
                print(f"{'='*70}")
                print(f"Best cost: {self.best_cost:.4f}")
                print(f"Best parameters:")
                for name, val in self._vector_to_dict(self.best_params).items():
                    print(f"  {name:20s}: {val:.4f}")
                print(f"{'='*70}\n")
            
            return {
                'best_params': self.best_params,
                'best_cost': self.best_cost,
                'all_results': self.trial_results,
                'optimization_result': result
            }
        
        except KeyboardInterrupt:
            print("\nOptimization interrupted")
            return {
                'best_params': self.best_params,
                'best_cost': self.best_cost,
                'all_results': self.trial_results,
                'optimization_result': None
            }
        
        finally:
            print("\nFinal cleanup...")
            self.cleanup_processes()
    
    def save_results(self, results: Dict, filename: str) -> None:
        """Save optimization results"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        names = self._names()
        
        # Load existing
        existing = {}
        if os.path.isfile(filename):
            try:
                with open(filename, 'r') as f:
                    existing = json.load(f) or {}
            except:
                existing = {}
        
        existing_trials = existing.get('trial_history', []) or []
        new_trials = results.get('all_results', []) or []
        
        # Merge and deduplicate
        combined = {}
        
        def to_key(tr):
            pd = tr.get('params') or {}
            if not all(n in pd for n in names):
                return tuple()
            vec = [float(pd[n]) for n in names]
            return self._canon_key(vec)
        
        for tr in existing_trials + new_trials:
            key = to_key(tr)
            if key == tuple():
                continue
            
            cost = tr.get('cost')
            if key not in combined or (cost is not None and 
                cost < combined[key].get('cost', float('inf'))):
                combined[key] = tr
        
        merged_trials = list(combined.values())
        
        # Find best
        best_cost = None
        best_params_dict = None
        
        for tr in merged_trials:
            if tr.get('failed'):
                continue
            c = tr.get('cost')
            pd = tr.get('params')
            if c is None or pd is None:
                continue
            
            if best_cost is None or c < best_cost:
                best_cost = c
                best_params_dict = pd
        
        output = {
            'best_cost': best_cost,
            'best_params': best_params_dict,
            'trial_history': merged_trials,
            'search_space': {
                'gs_kp_min': [0.3, 1.5],
                'gs_kd_max': [3.5, 6.0]
            },
            'fixed_params': {
                'att_kp_base': 'from config',
                'att_kd_base': 'from config',
                'rate_threshold': 'from config'
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Attitude Controller & Gain Scheduling Auto-Tuner"
    )
    parser.add_argument(
        "--use-history", 
        dest="use_history", 
        action="store_true", 
        default=True,
        help="Use previous results to warm start"
    )
    parser.add_argument(
        "--no-history",
        dest="use_history",
        action="store_false",
        help="Ignore history and start fresh"
    )
    parser.add_argument(
        "--max-seed-cost",
        type=float,
        default=50.0,
        help="Max cost from history to seed optimizer"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of optimization trials (default: 30, faster since only 2 params)"
    )
    
    args = parser.parse_args()
    
    config_file = "/home/pyro/ws_offboard_control/flight_data/config/control_params.json"
    flight_data_dir = "/home/pyro/ws_offboard_control/flight_data"
    
    tuner = AttitudeAutoTuner(
        config_file,
        flight_data_dir,
        reset_history=(not args.use_history),
        max_seed_cost=args.max_seed_cost
    )
    
    tuner.install_signal_handlers()
    
    # Run optimization
    results = tuner.optimize(n_calls=args.trials)
    
    # Save results
    tuner.save_results(results, tuner.results_file)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
