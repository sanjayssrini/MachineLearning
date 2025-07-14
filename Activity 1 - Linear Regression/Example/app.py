import joblib
import numpy as np

model = joblib.load('./model/model.pkl')

exp = float(input("Enter Years of Experience: "))

X_new = np.array(exp).reshape(1, -1)

pred = model.predict(X_new)
print(f"Predicted Salary: {pred[0]:.2f}")
