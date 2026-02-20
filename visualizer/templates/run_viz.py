#!/usr/bin/env python3
"""
Launcher for Parallel Hill Climber Visualizer
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    print("=" * 60)
    print("Parallel Hill Climber Visualizer")
    print("=" * 60)
    print("\nThis will launch both:")
    print("  1. The parallel hill climber evolution")
    print("  2. A real-time visualization of the population")
    print("\nOpen http://localhost:5004 in your browser")
    print("\nPress Ctrl+C to stop both processes\n")
    
    # Run the visualizer
    cmd = [sys.executable, "parallel_visualizer.py"]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)