from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import os
import numpy as np
import uuid

app = Flask(__name__) 

Expected = {
    "cylinders": {"min":3, "max": 8},
    "displacement": {"min":68.0, "max": 455.0},
    "horsepower": {"min":46.0, "max": 230.0},
    "weight": {"min":1613, "max": 5140},
    "acceleration": {"min":8.0, "max": 24.8},
    "model year": {"min":70, "max": 82},
    "origin": {"min":1, "max": 3}
}

model = load_model('Models/mpg_model.h5')

@app.route('/')
def index():
    return 'This is the homepage'

@app.route('/api', methods=['POST'])
def mpg_prediction():
    content = request.json
    errors = []
    for key in content:
        if key in Expected:
            expected_min = Expected[key]['min']
            expected_max = Expected[key]['max']
            value = content[key]
            if value < expected_min or value > expected_max:
                errors.append(f'Out of bounds: {key}, has value of: {value}, but it should be between {expected_min} and {expected_max}')
        else:
            errors.append(f'Unexpected field: {key}.')

    for key in Expected:
        if key not in content:
            errors.append(f'Missing value: {key}')
    if len(errors) < 1:
        x = np.zeros((1,7))

        x[0,0] = content['cylinders']
        x[0,1] = content['displacement']
        x[0,2] = content['horsepower']
        x[0,3] = content['weight']
        x[0,4] = content['acceleration']
        x[0,5] = content['model year']
        x[0,6] = content['origin']

        prediction = model.predict(x)
        mpg = float(prediction[0])
        response = { 'id': str(uuid.uuid4()), 'mpg': mpg, 'errors': errors}    
    else:
        response = {'id': str(uuid.uuid4()), 'errors': errors}
    
    return jsonify(response)

print(__name__)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)