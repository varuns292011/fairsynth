import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore")

def run_step(script, description):
    print(f"\n{'='*50}")
    print(f"RUNNING: {description}")
    print('='*50)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"ERROR in {script}")
        sys.exit(1)
    print(f"✅ {description} complete")

def run_step_with_arg(script, description, arg):
    print(f"\n{'='*50}")
    print(f"RUNNING: {description}")
    print('='*50)
    result = subprocess.run([sys.executable, script] + arg.split())
    if result.returncode != 0:
        print(f"ERROR in {script}")
        sys.exit(1)
    print(f"✅ {description} complete")

print("\n FAIRSYNTH PIPELINE STARTING ")
print("="*50)

csv_file = input("\nEnter CSV filename (or press Enter for insurance.csv): ").strip()
if csv_file == "":
    csv_file = "insurance.csv"

if not os.path.exists(csv_file):
    print(f"ERROR: {csv_file} not found!")
    sys.exit(1)

print(f"✅ Using dataset: {csv_file}")

run_step("fairness_analysis.py", "Fairness Analysis on Real Data")
run_step_with_arg("sdv_generator.py", "Generating Synthetic Data with SDV", csv_file)
run_step("compare_results.py", "Comparing Real vs Synthetic")
run_step_with_arg("discriminator.py", "Testing Synthetic Data Quality", f"{csv_file} {csv_file.replace('.csv', '_sdv_synthetic.csv')}")

print("\n FAIRSYNTH PIPELINE COMPLETE ")
print("="*50)