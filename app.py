#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user input from form
        user_input = [float(request.form[feature]) for feature in request.form]
        
        # Scale input data
        user_input_scaled = scaler.transform([user_input])
        
        # Make prediction
        prediction = model.predict(user_input_scaled)[0]
        
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return f"Error: {str(e)}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




