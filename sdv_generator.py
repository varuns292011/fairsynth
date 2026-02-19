import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

def generate_synthetic_sdv(csv_file, num_rows):
    print(f"Loading dataset: {csv_file}")
    data = pd.read_csv(csv_file)
    print(f"Original data shape: {data.shape}")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    print("Training synthesizer...")
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)
    print(f"Generating {num_rows} synthetic rows...")
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    output_path = csv_file.replace(".csv", "_sdv_synthetic.csv")
    synthetic_data.to_csv(output_path, index=False)
    print(f"Done! Saved to: {output_path}")

if len(sys.argv) == 3:
    generate_synthetic_sdv(sys.argv[1], int(sys.argv[2]))
else:
    csv_file = input("Enter CSV filename: ").strip()
    num_rows = int(input("Enter number of synthetic rows: ").strip())
    generate_synthetic_sdv(csv_file, num_rows)