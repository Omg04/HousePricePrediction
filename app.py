import pickle
import warnings
from flask import Flask, request, jsonify, render_template
import numpy as np

# Suppress sklearn unpickle version warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

# Expected features in the exact training column order
FEATURES = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    req_data = request.json.get('data', {})
    
    # Handle both dict with feature names and direct list of 13 values
    if isinstance(req_data, dict):
        normalized_data = {k.upper(): v for k, v in req_data.items()}
        input_values = [float(normalized_data[f]) for f in FEATURES]
    elif isinstance(req_data, list):
        input_values = [float(v) for v in req_data]
    else:
        return jsonify({'error': 'Invalid data format. Expected JSON object with 13 features.'}), 400

    input_data = np.array(input_values).reshape(1, -1)
    new_data = scalar.transform(input_data)
    output = regmodel.predict(new_data)
    prediction = round(float(output[0]), 2)
    print(f"Prediction: {prediction}k (${prediction * 1000:,.2f})")
    return jsonify({
        'Prediction': prediction,
        'Predicted_Price_USD': round(prediction * 1000, 2)
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features matching expected column order case-insensitively
    form_data = {k.upper(): v for k, v in request.form.items()}
    input_values = [float(form_data[f]) for f in FEATURES]
    
    final_input = scalar.transform(np.array(input_values).reshape(1, -1))
    output = regmodel.predict(final_input)
    predicted_price = round(float(output[0]), 2)
    return render_template("home.html", prediction_text=f"The predicted price is ${predicted_price * 1000:,.2f} ({predicted_price}k)")

if __name__ == "__main__":
    app.run(debug=True)