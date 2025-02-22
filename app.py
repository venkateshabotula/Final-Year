from flask import Flask, render_template, request, redirect, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import json
import re
import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from flask import Flask, render_template, request, url_for, send_from_directory, flash
import os
import shutil
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pdfrom 
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import io
import base64



from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import traceback



app = Flask(__name__)
app.secret_key = "water_reservoir"


water_demand_model = load_model("waterdemand\water_demand_model.h5")
with open("waterdemand\label_encoders.pkl", "rb") as f:
    water_demand_label_encoders = pickle.load(f)
with open("waterdemand\scaler (2).pkl", "rb") as f:
    water_demand_scaler = pickle.load(f)
@app.route('/')
def index():
    return redirect('/signup')


def save_to_json(data, filename="users.json"):
    try:
        with open(filename, "r") as f:
            users = json.load(f)
    except FileNotFoundError:
        users = []
    users.append(data)
    with open(filename, "w") as f:
        json.dump(users, f, indent=4)

def load_users(filename="users.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        # Validate Indian phone numbers
        if not re.fullmatch(r'[6-9]\d{9}', phone):
            return "Invalid Indian phone number! Must be 10 digits starting with 6-9."
        
        phone = f"+91{phone}"

        if password != confirm_password:
            return "Passwords do not match!"

        # Hash the password
        hashed_password = generate_password_hash(password)

        # Create a user object
        user = {
            "name": name,
            "username": username,
            "email": email,
            "phone": phone,
            "password": hashed_password,
            'latitude': latitude,
            'longitude': longitude,
        }

        save_to_json(user)
        return redirect('/login')

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Load users from JSON file
        users = load_users()

        # Find the user by username
        user = next((u for u in users if u["username"] == username), None)
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return redirect('/home')
        else:
            return "Invalid credentials!"

    return render_template('login.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')


app.route("/water-availability")
def water_availability():
    return render_template("water_availability.html")

@app.route("/water-demand")
def water_demand():
    return render_template("water_demand.html")


# Load the trained model
water_availability_model = load_model('wateravailability\lstm_water_availability_model.h5')

# Load label encoders
with open('wateravailability\label_encoders.pkl', 'rb') as f:
    water_avail_label_encoders = pickle.load(f)

# Load scalers
with open('wateravailability\scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)
water_avail_scaler_X, water_avail_scaler_y = scalers['scaler_X'], scalers['scaler_y']


@app.route('/water-availability', methods=['GET', 'POST'])
def water_availability():
    if request.method == 'POST':
        try:
            # Extract form data
            year = int(request.form['year'])
            district = request.form['district']
            name = request.form['name']
            month = request.form['month']
            rainfall = float(request.form['rainfall'])

            # Encode categorical inputs
            if district in water_avail_label_encoders['res_district'].classes_:
                encoded_district = water_avail_label_encoders['res_district'].transform([district])[0]
            else:
                return jsonify({'error': f'Invalid district: {district}'})

            if name in water_avail_label_encoders['res_name'].classes_:
                encoded_name = water_avail_label_encoders['res_name'].transform([name])[0]
            else:
                return jsonify({'error': f'Invalid reservoir name: {name}'})

            if month in water_avail_label_encoders['res_month'].classes_:
                encoded_month = water_avail_label_encoders['res_month'].transform([month])[0]
            else:
                return jsonify({'error': f'Invalid month: {month}'})

            # List of months (ensuring order)
            months_list = ["january", "february", "march", "april", "may", "june", 
                           "july", "august", "september", "october", "november", "december"]

            # Find index of selected month
            selected_month_index = months_list.index(month.lower())

            # Predict for all months up to the selected month
            predictions = []
            months_to_plot = []
            
            for i in range(selected_month_index + 1):  # Up to and including the selected month
                month_name = months_list[i]
                encoded_month = water_avail_label_encoders['res_month'].transform([month_name])[0]

                # Prepare input data
                input_data = np.array([[year, rainfall, encoded_district, encoded_month]])
                input_scaled = water_avail_scaler_X.transform(input_data)
                input_scaled = input_scaled.reshape((1, 1, input_scaled.shape[1]))  # Reshape for LSTM

                # Predict
                predicted_scaled = water_availability_model.predict(input_scaled)
                predicted_actual = water_avail_scaler_y.inverse_transform(predicted_scaled)

                predictions.append(float(predicted_actual[0][0]))
                months_to_plot.append(month_name.capitalize())

            # Generate the line plot
            plt.figure(figsize=(8, 5))
            plt.plot(months_to_plot, predictions, marker='o', linestyle='-', color='b')
            plt.xlabel('Month')
            plt.ylabel('Predicted Water Availability')
            plt.title(f'Water Availability Prediction for {district} - {year}')
            plt.grid(True)

            # Save the plot to a PNG image in memory
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()

            return render_template('water_availability.html', prediction=predictions[-1], graph_url=graph_url, 
                                   year=year, district=district, name=name, month=month, rainfall=rainfall)

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('water_availability.html', prediction=None, graph_url=None)
# @app.route('/water-availability', methods=['GET', 'POST'])
# def water_availability():
#     if request.method == 'POST':
#         try:
#             # Extract form data
#             year = int(request.form['year'])
#             district = request.form['district']
#             name = request.form['name']
#             month = request.form['month']
#             rainfall = float(request.form['rainfall'])

#             # Encode categorical inputs
#             if district in water_avail_label_encoders['res_district'].classes_:
#                 encoded_district = water_avail_label_encoders['res_district'].transform([district])[0]
#             else:
#                 return jsonify({'error': f'Invalid district: {district}'})

#             if name in water_avail_label_encoders['res_name'].classes_:
#                 encoded_name = water_avail_label_encoders['res_name'].transform([name])[0]
#             else:
#                 return jsonify({'error': f'Invalid reservoir name: {name}'})

#             if month in water_avail_label_encoders['res_month'].classes_:
#                 encoded_month = water_avail_label_encoders['res_month'].transform([month])[0]
#             else:
#                 return jsonify({'error': f'Invalid month: {month}'})

#             # Prepare input data
#             input_data = np.array([[year, rainfall, encoded_district, encoded_month]])
#             input_scaled = water_avail_scaler_X.transform(input_data)
#             input_scaled = input_scaled.reshape((1, 1, input_scaled.shape[1]))  # Reshape for LSTM

#             # Predict water availability
#             predicted_scaled = water_availability_model.predict(input_scaled)
#             predicted_actual = water_avail_scaler_y.inverse_transform(predicted_scaled)

#             return render_template('water_availability.html', prediction=float(predicted_actual[0][0]),year=year, district=district, name=name, 
#                            month=month, rainfall=rainfall)

#         except Exception as e:
#             return jsonify({'error': str(e)})

#     return render_template('water_availability.html', prediction=None)

import joblib
best_rf = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoders = joblib.load("models\label_encoders.pkl")
# Water Demand Prediction Page
@app.route('/water-demand', methods=['GET', 'POST'])
def water_demand_page():
    if request.method == 'POST':
        try:
            state = request.form['state']
            district = request.form['district']
            year = int(request.form['year'])
            selected_month = request.form['month']
            frl = float(request.form['frl'])
            liv_cap = float(request.form['liv_cap'])
            rainfall = float(request.form['rainfall'])
            population = int(request.form['population'])

            # Validate inputs against label encoders
            if state not in label_encoders["res_state"].classes_:
                return jsonify({'error': f'State "{state}" not found in training data'})
            if district not in label_encoders["res_district"].classes_:
                return jsonify({'error': f'District "{district}" not found in training data'})
            if selected_month not in label_encoders["res_month"].classes_:
                return jsonify({'error': f'Month "{selected_month}" not found in training data'})

            encoded_state = label_encoders["res_state"].transform([state])[0]
            encoded_district = label_encoders["res_district"].transform([district])[0]
            encoded_month = label_encoders["res_month"].transform([selected_month])[0]

            predictions = []
            population_labels = []

            for pop in range(population, population+400001, 100000):  # Vary population up to 300000
                prev_month_demand = predictions[-1] if predictions else 0
                input_data = np.array([[encoded_state, encoded_district, year, encoded_month, frl, liv_cap, rainfall, pop, 
                                        rainfall * pop, frl * liv_cap, prev_month_demand]])  # prev_month_demand placeholder
                input_scaled = scaler.transform(input_data)
                predicted_demand = best_rf.predict(input_scaled.reshape(1, -1))[0]  # Ensure proper shape
                predictions.append(predicted_demand)
                population_labels.append(str(pop))

            # Generate the plot
            plt.figure(figsize=(8, 4))
            plt.plot(population_labels, predictions, marker='o', linestyle='-')
            plt.xlabel('Population')
            plt.ylabel('Predicted Water Demand')
            plt.title(f'Water Demand Forecast for Population Growth')
            plt.grid(True)

            # Save the plot to a PNG image in memory
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()

            return render_template('water_demand.html', prediction=predictions[-1], graph_url=graph_url)
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('water_demand.html', prediction=None, graph_url=None)

# @app.route('/water-demand', methods=['GET', 'POST'])
# def water_demand_page():
#     if request.method == 'POST':
#         try:
#             state = request.form['state']
#             district = request.form['district']
#             year = int(request.form['year'])
#             selected_month = request.form['month']
#             frl = float(request.form['frl'])
#             liv_cap = float(request.form['liv_cap'])
#             rainfall = float(request.form['rainfall'])
#             population = int(request.form['population'])

#             encoded_state = label_encoders["res_state"].transform([state])[0]
#             encoded_district = label_encoders["res_district"].transform([district])[0]
#             encoded_month = label_encoders["res_month"].transform([selected_month])[0]

#             predictions = []
#             year_labels = []

#             for future_year in range(year, year + 5):  # Forecast for the next 5 years
#                 input_data = np.array([[encoded_state, encoded_district, future_year, encoded_month, frl, liv_cap, rainfall, population, 
#                                         rainfall * population, frl * liv_cap, 0]])  # prev_month_demand placeholder
#                 input_scaled = scaler.transform(input_data)
#                 predicted_demand = best_rf.predict(input_scaled)[0]
#                 predictions.append(predicted_demand)
#                 year_labels.append(str(future_year))

#             # Generate the plot
#             plt.figure(figsize=(8, 4))
#             plt.plot(year_labels, predictions, marker='o', linestyle='-')
#             plt.xlabel('Year')
#             plt.ylabel('Predicted Water Demand')
#             plt.title(f'Water Demand Forecast for {year} to {year + 4}')
#             plt.grid(True)

#             # Save the plot to a PNG image in memory
#             img = io.BytesIO()
#             plt.savefig(img, format='png')
#             img.seek(0)
#             graph_url = base64.b64encode(img.getvalue()).decode()

#             return render_template('water_demand.html', prediction=predictions[-1], graph_url=graph_url)
#         except Exception as e:
#             return jsonify({'error': str(e)})

#     return render_template('water_demand.html', prediction=None, graph_url=None)
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')

if __name__ == "__main__":
    app.run(debug=True)