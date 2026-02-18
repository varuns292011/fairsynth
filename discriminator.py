import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_discriminator(real_path, synthetic_path):
    real = pd.read_csv(real_path)
    synthetic = pd.read_csv(synthetic_path)
    real['is_real'] = 1
    synthetic['is_real'] = 0
    combined = pd.concat([real, synthetic], ignore_index=True)
    for col in combined.select_dtypes(include='object').columns:
        combined[col] = combined[col].astype('category').cat.codes
    X = combined.drop('is_real', axis=1)
    y = combined['is_real']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print("\nDISCRIMINATOR RESULTS")
    print("=" * 50)
    print(f"Detection Accuracy: {accuracy:.4f}")
    if accuracy > 0.85:
        print("VERDICT: Synthetic data is EASILY detectable")
    elif accuracy > 0.65:
        print("VERDICT: Synthetic data is SOMEWHAT detectable")
    else:
        print("VERDICT: Synthetic data is HARD to detect")

if len(sys.argv) == 3:
    run_discriminator(sys.argv[1], sys.argv[2])
else:
    real = input("Enter real CSV: ").strip()
    synthetic = input("Enter synthetic CSV: ").strip()
    run_discriminator(real, synthetic)