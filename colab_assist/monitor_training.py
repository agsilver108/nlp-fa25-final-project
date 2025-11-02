#!/usr/bin/env python3
"""
Real-time Log Monitor for Colab Training
Monitor training progress from VS Code terminal in real-time
"""

import os
import json
import time
import subprocess
import sys
from pathlib import Path

def setup_monitoring():
    """Setup real-time monitoring of Colab training output."""
    
    print("\n" + "="*70)
    print("🎯 COLAB TRAINING REAL-TIME MONITOR")
    print("="*70)
    print("\nThis script will monitor your Colab training in real-time.")
    print("\nInstructions:")
    print("1. Copy the code below to a Colab cell:")
    print("2. Run the Colab cell with GPU training")
    print("3. Download the log file when training completes")
    print("4. This script will monitor and display results")
    print("\n" + "="*70 + "\n")
    
    # Colab setup instructions
    colab_code = '''
# Copy and paste this into a Colab cell:

!git pull origin main  # Update repo
exec(open('colab_assist/colab_streaming_training.py').read())
    '''
    
    print("📋 COLAB CELL CODE:")
    print("-" * 70)
    print(colab_code)
    print("-" * 70)
    
    print("\n✅ After training starts:")
    print("   1. Download 'colab_training_stream.log' from Colab")
    print("   2. Save it to: colab_training_stream.log (current directory)")
    print("   3. This script will then monitor and display it")
    print("\nPress Enter when you're ready to start monitoring...")
    input()

def monitor_log_file(log_file="colab_training_stream.log"):
    """Monitor the log file for updates in real-time."""
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        print("   Make sure you downloaded it from Colab!")
        return
    
    print(f"\n✅ Found log file: {log_file}")
    print("📊 Monitoring training progress...\n")
    
    # Start monitoring
    last_size = 0
    last_lines = 0
    check_interval = 2  # Check every 2 seconds
    
    try:
        while True:
            if os.path.exists(log_file):
                current_size = os.path.getsize(log_file)
                
                # Read file
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    current_lines = len(lines)
                
                # Check for new content
                if current_size > last_size or current_lines > last_lines:
                    # Print new lines
                    start_line = last_lines
                    for i, line in enumerate(lines[start_line:], start=start_line):
                        print(line.rstrip())
                    
                    last_size = current_size
                    last_lines = current_lines
                
                # Check if training is complete
                if lines and "Training complete!" in lines[-1]:
                    print("\n" + "="*70)
                    print("✅ TRAINING COMPLETE!")
                    print("="*70)
                    
                    # Try to load and display results
                    results_file = "colab_training_results.json"
                    if os.path.exists(results_file):
                        print("\n📊 FINAL RESULTS:")
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                        
                        print(f"\nBaseline EM:      {results['baseline']['exact_match']:.4f}")
                        print(f"Baseline F1:      {results['baseline']['f1']:.4f}")
                        print(f"Cartography EM:   {results['cartography']['exact_match']:.4f}")
                        print(f"Cartography F1:   {results['cartography']['f1']:.4f}")
                        print(f"\nEM Improvement:   {results['improvement']['em_diff']:+.4f}")
                        print(f"F1 Improvement:   {results['improvement']['f1_diff']:+.4f}")
                        print(f"\nTotal Time:       {results['training_time_minutes']:.1f} minutes")
                    
                    break
                
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")

def run_monitoring_cli():
    """Simple CLI for monitoring."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Colab Training Output')
    parser.add_argument('--log-file', default='colab_training_stream.log', 
                       help='Path to log file to monitor')
    parser.add_argument('--setup', action='store_true', 
                       help='Show setup instructions')
    
    args = parser.parse_args()
    
    if args.setup or not os.path.exists(args.log_file):
        setup_monitoring()
    
    monitor_log_file(args.log_file)

if __name__ == "__main__":
    run_monitoring_cli()
