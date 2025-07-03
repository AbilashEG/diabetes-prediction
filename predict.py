import sys
import xgboost as xgb
import numpy as np

def predict_diabetes(input_data):
    # Load saved model
    model = xgb.Booster()
    model.load_model("model/diabetes_model.bst")

    # Convert input string to float array
    data = np.array([float(x) for x in input_data.split(",")]).reshape(1, -1)
    dmatrix = xgb.DMatrix(data)

    # Predict probability
    prob = model.predict(dmatrix)[0]
    print(f"Prediction probability: {prob:.4f}")

    if prob > 0.5:
        print("Result: Diabetic")
    else:
        print("Result: Not Diabetic")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py --input \"val1,val2,...,val8\"")
        sys.exit(1)

    input_str = sys.argv[1]
    predict_diabetes(input_str)
