import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_fairness(data, label):
    data = data.copy()
    data = data.fillna(data.mean(numeric_only=True))
    if 'sex' in data.columns:
        data['sex'] = data['sex'].map({'male': 0, 'female': 1})
    if 'smoker' in data.columns:
        data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
    if 'region' in data.columns:
        data = pd.get_dummies(data, columns=['region'], drop_first=True)
    if 'charges' in data.columns:
        target = 'charges'
    else:
        target = data.columns[-1]
    if 'age' not in data.columns:
        print(f"\n{label} - No age column found")
        return
    data['age_group'] = pd.cut(data['age'], bins=[0, 30, 50, 100], labels=['young', 'middle', 'older'])
    data = data.dropna()
    X = data.drop([target, 'age_group'], axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n{label}")
    print("=" * 50)
    print(f"Overall R²: {r2_score(y_test, predictions):.4f}")
    print(f"Overall MAE: {mean_absolute_error(y_test, predictions):.2f}")
    test_data = X_test.copy()
    test_data['actual'] = y_test.values
    test_data['predicted'] = predictions
    test_data['age_group'] = data.loc[X_test.index, 'age_group'].values
    for group in ['young', 'middle', 'older']:
        group_data = test_data[test_data['age_group'] == group]
        if len(group_data) > 1:
            r2 = r2_score(group_data['actual'], group_data['predicted'])
            mae = mean_absolute_error(group_data['actual'], group_data['predicted'])
            print(f"{group.upper()}: R²={r2:.4f}, MAE={mae:.2f}")

real_data = pd.read_csv("insurance.csv")
synthetic_file = sys.argv[1] if len(sys.argv) > 1 else input("Enter synthetic CSV filename: ").strip()
synthetic_data = pd.read_csv(synthetic_file)

evaluate_fairness(real_data, "REAL DATA RESULTS")
evaluate_fairness(synthetic_data, "SYNTHETIC DATA RESULTS")