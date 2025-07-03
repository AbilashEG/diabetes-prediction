import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Load dataset
df = pd.read_csv("data/diabetes.csv", header=None)

# Rename columns for clarity (optional)
df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
              'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Split features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Prepare DMatrices for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.1,
    "seed": 42
}

# Train model
model = xgb.train(params, dtrain, num_boost_round=100)

# Predict on test set
y_pred_prob = model.predict(dtest)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")

# Save model
os.makedirs("model", exist_ok=True)
model.save_model("model/diabetes_model.bst")
print("✅ Model trained and saved!")

