document.addEventListener('DOMContentLoaded', function() {
    fetch('/static/column_names.json')
        .then(response => response.json())
        .then(data => {
            const formFields = document.getElementById('additionalFields');
            const excludedColumns = [
                'Last_Service_Date', 'Warranty_Expiry_Date', 'Vehicle_Model',
                'Maintenance_History', 'Transmission_Type', 'Tire_Condition',
                'Brake_Condition', 'Battery_Status', 'Mileage', 'Reported_Issues',
                'Vehicle_Age', 'Engine_Size', 'Odometer_Reading', 'Service_History',
                'Accident_History', 'Fuel_Type', 'Fuel_Efficiency', 'Owner_Type'
            ];

            // Add other fields not in excludedColumns
            data.forEach(col => {
                if (!excludedColumns.includes(col)) {
                    const div = document.createElement('div');
                    div.classList.add('form-group');
                    div.innerHTML = `<label>${col}</label><input type="text" name="${col}" required>`;
                    formFields.appendChild(div);
                }
            });
        });
});

document.getElementById('predictionForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const formObject = {};
    formData.forEach((value, key) => {
        formObject[key] = value;
    });

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formObject),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').innerText = `Error: ${data.error}`;
        } else {
            document.getElementById('result').innerText = data.prediction;
        }
    })
    .catch(error => {
        document.getElementById('result').innerText = `Error: ${error.message}`;
    });
});
