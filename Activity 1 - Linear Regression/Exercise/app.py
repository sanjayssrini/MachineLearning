import joblib
import numpy as np

model = joblib.load('./model/taxi_fare_model.joblib')

trip_miles = float(input("Enter trip miles: "))
trip_seconds = float(input("Enter trip duration in seconds: "))

speed = trip_miles / (trip_seconds / 3600)

X_new = np.array([[trip_miles, trip_seconds, speed]])

predicted_fare = model.predict(X_new)
print(f"Predicted Fare: ${predicted_fare[0]:.2f}")
