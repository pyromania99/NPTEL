import os
import json
from auto_tuner import ControllerAutoTuner

# Paths
flight_data_dir = "/home/pyro/ws_offboard_control/flight_data"
config_file = os.path.join(flight_data_dir, "config/control_params.json")
results_file = os.path.join(flight_data_dir, "outputs/tuning_results.json")

# Load previous results
with open(results_file, 'r') as f:
    db = json.load(f)

trials = db.get('trial_history', [])

# Initialize tuner

tuner = ControllerAutoTuner(config_file, flight_data_dir)
tuner.install_signal_handlers()

new_trials = []

for i, trial in enumerate(trials):
    params_dict = trial.get('params')
    if not params_dict:
        continue
    print(f"\n=== Rerunning trial {i+1}/{len(trials)} ===")
    param_vec = tuner._dict_to_vector(params_dict)
    tuner.update_config(param_vec)
    try:
        log_file = tuner.run_flight_test(duration=30.0)
        cost = tuner.analyze_performance(log_file)
        print(f"New cost: {cost:.4f}")
        new_trials.append({
            'params': params_dict,
            'cost': cost,
            'log_file': log_file
        })
    except Exception as e:
        print(f"Trial failed: {e}")
        new_trials.append({
            'params': params_dict,
            'cost': 1000.0,
            'log_file': None,
            'failed': True,
            'error': str(e)
        })

# Save new results
new_db = db.copy()
new_db['trial_history'] = new_trials

# Recompute best
best_cost = None
best_params = None
for t in new_trials:
    if t.get('failed'):
        continue
    c = t.get('cost')
    pd = t.get('params')
    if c is None or pd is None:
        continue
    try:
        c = float(c)
    except Exception:
        continue
    if best_cost is None or c < best_cost:
        best_cost = c
        best_params = pd
new_db['best_cost'] = best_cost
new_db['best_params'] = best_params

with open(results_file, 'w') as f:
    json.dump(new_db, f, indent=2)

print("All trials rerun and results updated.")
