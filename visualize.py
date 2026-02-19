import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings("ignore")

def create_visualizations(real_path, synthetic_path):
    real = pd.read_csv(real_path)
    synthetic = pd.read_csv(synthetic_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FairSynth: Real vs Synthetic Data Analysis', fontsize=16, fontweight='bold')

    # Plot 1 — Age distribution
    axes[0,0].hist(real['age'], alpha=0.6, label='Real', color='blue', bins=20)
    axes[0,0].hist(synthetic['age'], alpha=0.6, label='Synthetic', color='red', bins=20)
    axes[0,0].set_title('Age Distribution')
    axes[0,0].set_xlabel('Age')
    axes[0,0].legend()

    # Plot 2 — Charges distribution
    axes[0,1].hist(real['charges'], alpha=0.6, label='Real', color='blue', bins=20)
    axes[0,1].hist(synthetic['charges'], alpha=0.6, label='Synthetic', color='red', bins=20)
    axes[0,1].set_title('Charges Distribution')
    axes[0,1].set_xlabel('Charges ($)')
    axes[0,1].legend()

    # Plot 3 — BMI distribution
    axes[1,0].hist(real['bmi'], alpha=0.6, label='Real', color='blue', bins=20)
    axes[1,0].hist(synthetic['bmi'], alpha=0.6, label='Synthetic', color='red', bins=20)
    axes[1,0].set_title('BMI Distribution')
    axes[1,0].set_xlabel('BMI')
    axes[1,0].legend()

    # Plot 4 — Detection accuracy by rows
    rows = [500, 1000, 2000, 2500, 3000, 3500, 4000]
    accuracy = [0.7880, 0.8077, 0.8069, 0.7982, 0.8053, 0.8161, 0.8221]
    axes[1,1].plot(rows, accuracy, marker='o', color='green', linewidth=2)
    axes[1,1].axhline(y=0.65, color='red', linestyle='--', label='Usability threshold')
    axes[1,1].set_title('Detection Accuracy vs Rows Generated')
    axes[1,1].set_xlabel('Synthetic Rows')
    axes[1,1].set_ylabel('Detection Accuracy')
    axes[1,1].legend()

    plt.tight_layout()
    plt.savefig('fairsynth_results.png', dpi=150)
    print("Chart saved as fairsynth_results.png")
    plt.show()

real = sys.argv[1] if len(sys.argv) > 1 else input("Enter real CSV: ").strip()
synthetic = sys.argv[2] if len(sys.argv) > 2 else input("Enter synthetic CSV: ").strip()
create_visualizations(real, synthetic)