import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
n_samples = 1000

tenure = np.random.randint(1, 72, size=n_samples)
monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
total_charges = tenure * monthly_charges + np.random.normal(0, 100, size=n_samples)

churn_prob = 1 / (1 + np.exp(0.05 * (tenure) - 0.03 * (monthly_charges+1)))
churn = (np.random.rand(n_samples) < churn_prob).astype(int)

# Create a DataFrame
df = pd.DataFrame({
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'churn': churn
})

# Split the data into features and target
X = df[['tenure', 'monthly_charges', 'total_charges']]
y = df['churn']

# Initialize the model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'churn_model.pkl')