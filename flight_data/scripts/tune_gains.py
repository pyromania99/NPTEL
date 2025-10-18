#!/usr/bin/env python3
"""
Control Parameter Tuning Utility
Easy way to modify control gains without rebuilding
"""

import json
import sys
import os

CONFIG_FILE = "/home/pyro/ws_offboard_control/flight_data/config/control_params.json"

def load_config():
    """Load current configuration"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file not found: {CONFIG_FILE}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config file: {e}")
        return None

def save_config(config):
    """Save configuration with nice formatting"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {CONFIG_FILE}")

def show_current_config():
    """Display current configuration"""
    config = load_config()
    if config is None:
        return
    
    print("Current Control Parameters:")
    print("=" * 40)
    
    for section, params in config.items():
        if section.startswith("_"):  # Skip comments
            continue
        print(f"\n{section.upper()}:")
        for key, value in params.items():
            print(f"  {key}: {value}")

def set_parameter(section, key, value):
    """Set a specific parameter"""
    config = load_config()
    if config is None:
        return
    
    if section not in config:
        print(f"Unknown section: {section}")
        print(f"Available sections: {[s for s in config.keys() if not s.startswith('_')]}")
        return
    
    if key not in config[section]:
        print(f"Unknown parameter '{key}' in section '{section}'")
        print(f"Available parameters: {list(config[section].keys())}")
        return
    
    try:
        # Convert to float
        config[section][key] = float(value)
        save_config(config)
        print(f"Set {section}.{key} = {value}")
    except ValueError:
        print(f"Invalid value '{value}' - must be a number")

def quick_tune_attitude(kp, kd=None):
    """Quick tune attitude gains"""
    config = load_config()
    if config is None:
        return
    
    config["attitude"]["kp"] = float(kp)
    if kd is not None:
        config["attitude"]["kd"] = float(kd)
    
    save_config(config)
    print(f"Attitude gains updated: Kp={kp}" + (f", Kd={kd}" if kd else ""))

def quick_tune_ndi(kp, kd=None):
    """Quick tune NDI rate gains"""
    config = load_config()
    if config is None:
        return
    
    config["ndi_rate"]["kp"] = float(kp)
    if kd is not None:
        config["ndi_rate"]["kd"] = float(kd)
    
    save_config(config)
    print(f"NDI rate gains updated: Kp={kp}" + (f", Kd={kd}" if kd else ""))

def main():
    if len(sys.argv) < 2:
        print("Control Parameter Tuning Utility")
        print("Usage:")
        print("  python3 tune_gains.py show                    # Show current config")
        print("  python3 tune_gains.py set <section> <key> <value>  # Set parameter")
        print("  python3 tune_gains.py att <kp> [kd]           # Quick attitude tune")
        print("  python3 tune_gains.py ndi <kp> [kd]           # Quick NDI tune")
        print()
        print("Examples:")
        print("  python3 tune_gains.py show")
        print("  python3 tune_gains.py set attitude kp 5.0")
        print("  python3 tune_gains.py att 4.5 0.1")
        print("  python3 tune_gains.py ndi 0.15")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "show":
        show_current_config()
    
    elif cmd == "set":
        if len(sys.argv) != 5:
            print("Usage: python3 tune_gains.py set <section> <key> <value>")
            return
        section, key, value = sys.argv[2], sys.argv[3], sys.argv[4]
        set_parameter(section, key, value)
    
    elif cmd == "att":
        if len(sys.argv) < 3:
            print("Usage: python3 tune_gains.py att <kp> [kd]")
            return
        kp = sys.argv[2]
        kd = sys.argv[3] if len(sys.argv) > 3 else None
        quick_tune_attitude(kp, kd)
    
    elif cmd == "ndi":
        if len(sys.argv) < 3:
            print("Usage: python3 tune_gains.py ndi <kp> [kd]")
            return
        kp = sys.argv[2]
        kd = sys.argv[3] if len(sys.argv) > 3 else None
        quick_tune_ndi(kp, kd)
    
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()