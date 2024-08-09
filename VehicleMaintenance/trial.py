import pickle
import pandas as pd

# Load the model
model_path = r"C:/Users/Guest_User/Documents/VehicleMaintenance/static/best_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Create a sample input matching the expected format
sample_input = {
    "Mileage": 31653,
    "Reported_Issues": 2,
    "Vehicle_Age": 1,
    "Engine_Size": 800,
    "Odometer_Reading": 70954,
    "Service_History": 5,
    "Accident_History": 3,
    "Last_Service_Year": 2023,
    "Last_Service_Month": 8,
    "Last_Service_Day": 12,
    "Last_Service_DayOfWeek": 5,
    "Warranty_Expiry_Year": 2024,
    "Warranty_Expiry_Month": 9,
    "Warranty_Expiry_Day": 5,
    "Warranty_Expiry_DayOfWeek": 3,
    "Vehicle_Model_Bus": 0,
    "Vehicle_Model_Car": 0,
    "Vehicle_Model_Motorcycle": 0,
    "Vehicle_Model_SUV": 0,
    "Vehicle_Model_Truck": 1,
    "Vehicle_Model_Van": 0,
    "Maintenance_History_Average": 1,
    "Maintenance_History_Good": 0,
    "Maintenance_History_Poor": 0,
    "Transmission_Type_Automatic": 1,
    "Transmission_Type_Manual": 0,
    "Tire_Condition_Good": 0,
    "Tire_Condition_New": 1,
    "Tire_Condition_Worn Out": 0,
    "Brake_Condition_Good": 1,
    "Brake_Condition_New": 0,
    "Brake_Condition_Worn Out": 0,
    "Battery_Status_Good": 0,
    "Battery_Status_New": 1,
    "Battery_Status_Weak": 0
}

# Convert to DataFrame
input_df = pd.DataFrame([sample_input])

# Predict
prediction = model.predict(input_df)
result = 'Needs Maintenance' if prediction[0] else 'No Maintenance Needed'
print(f"Prediction: {result}")