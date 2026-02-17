import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
data = pd.read_csv("insurance.csv")
data['sex'] = data['sex'].map({'male': 0, 'female': 1})
data['smoker'] = data['smoker'].map({'no': 0, 'yes': 1})
data = pd.get_dummies(data, columns=['region'], drop_first=True)
data['age_group'] = pd.cut(data['age'], bins=[0, 30, 50, 100], labels=['young', 'middle', 'older'])

X = data.drop(['charges', 'age_group'], axis=1)
y = data['charges']

# Store results across 10 runs
results = {'young': [], 'middle': [], 'older': []}

print("Running 10 different train/test splits...\n")

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    test_data = X_test.copy()
    test_data['actual'] = y_test.values
    test_data['predicted'] = predictions
    test_data['age_group'] = data.loc[X_test.index, 'age_group'].values
    
    for group in ['young', 'middle', 'older']:
        group_data = test_data[test_data['age_group'] == group]
        if len(group_data) > 0:
            r2 = r2_score(group_data['actual'], group_data['predicted'])
            results[group].append(r2)

# Print averaged results
print("AVERAGED RESULTS ACROSS 10 RUNS")
print("=" * 50)
for group in ['young', 'middle', 'older']:
    avg_r2 = np.mean(results[group])
    std_r2 = np.std(results[group])
    print(f"{group.upper()}: R² = {avg_r2:.4f} (±{std_r2:.4f})")