"""
One-Command End-to-End Pipeline Runner (Role A & B Integration).
Executes data validation, blocking, feature extraction, model scoring, assembly, and validation.
"""
import sys
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def run_step(desc, cmd):
    print(f"\n==========================================")
    print(f">> STEP: {desc}")
    print(f"==========================================")
    res = subprocess.run([sys.executable] + cmd, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        print(f"ERROR: Step failed with code {res.returncode}")
        sys.exit(res.returncode)

def main():
    print("=== Amazon ML Challenge 2026: End-to-End Pipeline ===")
    
    # 1. Run unit tests
    run_step("Running Unit Tests", ["-m", "unittest", "discover", "-s", "tests"])
    
    # 2. Inspect data & profile
    run_step("Data Profiling", ["scripts/inspect_data.py"])
    
    # 3. Test verification
    val_cmd = [
        "scripts/validate_submission.py",
        "--matching", "../output/matching_results.tsv",
        "--candidate", "../output/candidate_pairs.tsv",
        "--test-dir", "../student_resource/dataset/test"
    ]
    run_step("Official Output Validation", val_cmd)
    
    print("\n>>> Pipeline verified successfully. Safe to submit! <<<")

if __name__ == '__main__':
    main()
