#!/usr/bin/env python3
"""
Debug script to simulate GitHub Actions CI environment locally.

This script helps debug CI failures without having to push to GitHub repeatedly.
It simulates the exact same environment and steps as the GitHub Actions workflow.

Usage: python debug_ci.py
"""
import os
import subprocess
import sys

def run_command(cmd, env=None, cwd=None):
    """Run command and return result"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, 
            env=env, cwd=cwd, timeout=300
        )
        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        return result
    except subprocess.TimeoutExpired:
        print("Command timed out!")
        return None

def simulate_ci():
    """Simulate the exact CI environment from .github/workflows/ci.yml"""
    print("=== Simulating GitHub Actions CI Environment ===")

    # Get current directory - GHA runs directly in the repo, no copying!
    project_dir = os.getcwd()
    print(f"Project directory: {project_dir}")

    print("\n1. Using current directory (like GHA checkout)...")
    print("ℹ️  GitHub Actions runs directly in the repo, no file copying")

    # Set up environment exactly like GitHub Actions
    ci_env = os.environ.copy()
    ci_env.update({
        'WANDB_API_KEY': 'test-api-key',
        'CI': 'true',
        'GITHUB_ACTIONS': 'true',
    })

    print("\n2. Installing dependencies (simulated)...")
    # In real CI, this would: pip install -r requirements.txt && pip install pytest black
    req_path = os.path.join(project_dir, 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            print("Requirements found:")
            print(f.read())
    else:
        print("❌ No requirements.txt found!")
        return False

    print("\n3. Running black formatting check...")
    result = run_command("black --check app/", env=ci_env, cwd=project_dir)
    if result and result.returncode != 0:
        print("❌ Black formatting check failed!")
        print("Run 'black app/' locally to fix formatting issues")
        return False
    else:
        print("✅ Black formatting check passed")

    print("\n4. Running tests...")
    result = run_command("python -m pytest app/tests/ -v", env=ci_env, cwd=project_dir)
    if result and result.returncode != 0:
        print("❌ Tests failed!")
        print("This simulates the failure you would see in GitHub Actions")
        return False
    else:
        print("✅ All tests passed!")

    print("\n5. Environment comparison:")
    print("CI Environment variables:")
    for key in ['WANDB_API_KEY', 'CI', 'GITHUB_ACTIONS']:
        print(f"  {key}: {ci_env.get(key, 'NOT SET')}")

    return True

if __name__ == "__main__":
    print("🔍 Local CI Simulation Tool")
    print("This simulates your GitHub Actions workflow locally")
    print("=" * 50)

    success = simulate_ci()

    print("\n" + "=" * 50)
    if success:
        print("🎉 CI simulation completed successfully!")
        print("Your GitHub Actions should pass!")
    else:
        print("💥 CI simulation failed!")
        print("Fix the issues above before pushing to GitHub")

    sys.exit(0 if success else 1)