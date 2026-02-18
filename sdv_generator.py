import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

def generate_synthetic_sdv(csv_file):
    print(f"Loading dataset: {csv_file}")
    data = pd.read_csv(csv_file)
    print(f"Original data shape: {data.shape}")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    print("Training synthesizer...")
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    print("Generating 3000 synthetic rows...")
    synthetic_data = synthesizer.sample(num_rows=3000)
    output_path = csv_file.replace(".csv", "_sdv_synthetic.csv")
    synthetic_data.to_csv(output_path, index=False)
    print(f"Done! Saved to: {output_path}")

csv_file = sys.argv[1] if len(sys.argv) > 1 else input("Enter your CSV filename: ").strip()
generate_synthetic_sdv(csv_file)