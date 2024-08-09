from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import json
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load the model
model_path = r"C:/Users/Guest_User/Documents/VehicleMaintenance/static/b_model.keras"
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    model = None
    print(f"Error loading the model from {model_path}: {e}")

# Load the column names
column_names_path = r"C:/Users/Guest_User/Documents/VehicleMaintenance/static/column_names.json"
try:
    with open(column_names_path, "r") as json_file:
        column_names = json.load(json_file)
    print(f"Column names loaded successfully from {column_names_path}")
except Exception as e:
    column_names = []
    print(f"Error loading column names from {column_names_path}: {e}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'})
    try:
        data = request.json
        input_data = []
        for col in column_names:
            if col == 'Last_Service_Date' or col == 'Warranty_Expiry_Date':
                date_str = data.get(col)
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                input_data.extend([date_obj.year, date_obj.month, date_obj.day])
            elif col == 'Vehicle_Model':
                vehicle_model = data.get('Vehicle_Model')
                vehicle_model_columns = ['Vehicle_Model_Bus', 'Vehicle_Model_Car', 'Vehicle_Model_Motorcycle', 'Vehicle_Model_SUV', 'Vehicle_Model_Truck', 'Vehicle_Model_Van']
                for vm in vehicle_model_columns:
                    input_data.append(1 if vm.split('_')[-1] == vehicle_model else 0)
            elif col == 'Maintenance_History':
                maintenance_history = data.get('Maintenance_History')
                maintenance_history_columns = ['Maintenance_History_Average', 'Maintenance_History_Good', 'Maintenance_History_Poor']
                for mh in maintenance_history_columns:
                    input_data.append(1 if mh.split('_')[-1] == maintenance_history else 0)
            elif col == 'Transmission_Type':
                transmission_type = data.get('Transmission_Type')
                transmission_type_columns = ['Transmission_Type_Automatic', 'Transmission_Type_Manual']
                for tt in transmission_type_columns:
                    input_data.append(1 if tt.split('_')[-1] == transmission_type else 0)
            elif col == 'Tire_Condition':
                tire_condition = data.get('Tire_Condition')
                tire_condition_columns = ['Tire_Condition_Good', 'Tire_Condition_New', 'Tire_Condition_Worn Out']
                for tc in tire_condition_columns:
                    input_data.append(1 if tc.split('_')[-1] == tire_condition else 0)   
            elif col == 'Brake_Condition':
                brake_condition = data.get('Brake_Condition')
                brake_condition_columns = ['Brake_Condition_Good', 'Brake_Condition_New', 'Brake_Condition_Worn Out']
                for bc in brake_condition_columns:
                    input_data.append(1 if bc.split('_')[-1] == brake_condition else 0)   
            elif col == 'Battery_Status':
                battery_status = data.get('Battery_Status')
                battery_status_columns = ['Battery_Status_Good', 'Battery_Status_New', 'Battery_Status_Weak']
                for bs in battery_status_columns:
                    input_data.append(1 if bs.split('_')[-1] == battery_status else 0)
            elif col == 'Fuel_Type':
                fuel_type = data.get('Fuel_Type')
                fuel_type_columns = ['Fuel_Type_Diesel', 'Fuel_Type_Electric', 'Fuel_Type_Petrol']
                for ft in fuel_type_columns:
                    input_data.append(1 if ft.split('_')[-1] == fuel_type else 0)
            elif col == 'Owner_Type':
                owner_type = data.get('Owner_Type')
                owner_type_columns = ['Owner_Type_First', 'Owner_Type_Second', 'Owner_Type_Third']
                for ot in owner_type_columns:
                    input_data.append(1 if ot.split('_')[-1] == owner_type else 0)     
            else:
                input_data.append(data.get(col, 0))
        
        # Convert input data to float
        input_data = np.array(input_data, dtype=float).reshape(1, -1)
        
        prediction = model.predict(input_data)
        result = 'Needs Maintenance' if prediction[0][0] > 0.5 else 'No Maintenance Needed'
        
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
