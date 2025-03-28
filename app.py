#!/usr/bin/env python
# coding: utf-8

# In[1]:


import flask 


# In[2]:


from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Updated Predict Route (Add this in place of the existing one)
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('error.html', message="Error: Model or Scaler not loaded!")

    try:
        # Extract user input safely
        user_input = []
        for feature in request.form:
            value = request.form[feature].strip()
            if value:  # Ensure value is not empty
                user_input.append(float(value))

        # Check for empty input
        if len(user_input) == 0:
            return render_template('error.html', message="Error: No input provided! Please enter valid data.")

        # Scale input data
        user_input_scaled = scaler.transform([user_input])
        
        # Make prediction
        prediction = model.predict(user_input_scaled)[0]
        
        return render_template('result.html', prediction=prediction)
    except ValueError:
        return render_template('error.html', message="Error: Invalid input format! Please enter numeric values.")
    except Exception as e:
        return render_template('error.html', message=f"Error: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




