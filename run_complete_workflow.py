"""
Complete workflow script: Record session, analyze, and generate report
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {command}\n")
    
    try:
        result = subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return False

def main():
    print("\n" + "="*70)
    print("COMPLETE POSE TRACKING WORKFLOW")
    print("="*70)
    print("\nThis script will guide you through:")
    print("1. System check")
    print("2. Camera test")
    print("3. Recording session")
    print("4. Analysis and visualization")
    print("5. Report generation")
    print("\n" + "="*70)
    
    input("\nPress ENTER to start...")
    
    steps = [
        ("python check_system.py", "System Specifications Check"),
        ("python test_camera.py", "Camera Access Test (Press 'q' to continue)"),
        ("python main.py", "Record Pose Tracking Session (Press SPACE to start/stop, 'q' to quit)"),
        ("python analyze_session.py", "Analyze Session and Generate Plots"),
        ("python generate_report.py", "Generate Session Report")
    ]
    
    for i, (command, description) in enumerate(steps, 1):
        print(f"\n\n{'='*70}")
        print(f"STEP {i}/{len(steps)}: {description}")
        print(f"{'='*70}")
        
        if i == 2:
            print("\nNote: You need to manually press 'q' in the camera window to continue")
            input("Press ENTER when camera test window is closed...")
            continue
        elif i == 3:
            print("\nNote: Recording session - Press SPACE to start/stop recording")
            print("Move your arms around during recording")
            input("Press ENTER to start recording session...")
        
        success = run_command(command, description)
        
        if not success and i == 3:
            print("\nRecording session ended or cancelled")
            response = input("Continue with analysis? (y/n): ")
            if response.lower() != 'y':
                print("\nWorkflow cancelled")
                return
        elif not success:
            print(f"\nStep {i} failed. Continue anyway? (y/n): ", end='')
            response = input()
            if response.lower() != 'y':
                print("\nWorkflow cancelled")
                return
    
    print("\n\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print("\nAll outputs saved to:")
    print("  - Videos: outputs/videos/")
    print("  - Data: outputs/data/")
    print("  - Plots: outputs/plots/")
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(0)
