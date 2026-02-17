import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
data = pd.read_csv("insurance.csv")

# Convert categorical to numeric
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data = pd.get_dummies(data, columns=['region'], drop_first=True)

# Create age groups for fairness analysis
data['age_group'] = pd.cut(data['age'], bins=[0, 30, 50, 100], labels=['young', 'middle', 'older'])

# Features and target
X = data.drop(['charges', 'age_group'], axis=1)
y = data['charges']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Overall performance
print("OVERALL MODEL PERFORMANCE")
print("=" * 50)
print(f"R² Score: {r2_score(y_test, predictions):.4f}")
print(f"Mean Absolute Error: ${mean_absolute_error(y_test, predictions):.2f}")

# Fairness analysis by age group
print("\n\nFAIRNESS ANALYSIS BY AGE GROUP")
print("=" * 50)

test_data = X_test.copy()
test_data['actual'] = y_test.values
test_data['predicted'] = predictions
test_data['age_group'] = data.loc[X_test.index, 'age_group'].values

for group in ['young', 'middle', 'older']:
    group_data = test_data[test_data['age_group'] == group]
    if len(group_data) > 0:
        mae = mean_absolute_error(group_data['actual'], group_data['predicted'])
        r2 = r2_score(group_data['actual'], group_data['predicted'])
        print(f"\n{group.upper()} (n={len(group_data)})")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: ${mae:.2f}")