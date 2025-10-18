#!/usr/bin/env python3
"""Legacy wrapper for backward compatibility - redirects to organized structure."""

import os
import sys
import subprocess

def main():
    # Path to the organized plotting script
    script_dir = os.path.join(os.path.dirname(__file__), 'flight_data', 'scripts')
    actual_script = os.path.join(script_dir, 'generate_plots.py')
    
    if not os.path.exists(actual_script):
        print(f"Error: Could not find plotting script at {actual_script}")
        print("Run from the workspace root directory containing flight_data/")
        sys.exit(1)
    
    print(f"[INFO] Redirecting to organized plotting script: {actual_script}")
    
    # Pass all arguments to the actual script, ensuring outputs go to organized directory
    cmd = [sys.executable, actual_script, '--output-dir', 'flight_data/outputs'] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running plotting script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()