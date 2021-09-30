import os
from tensorflow.keras.models import load_model
import numpy as np
import requests


model = load_model('Models/mpg_model.h5')

x = np.zeros((1,7))

x[0,0] = 8
x[0,1] = 400
x[0,2] = 80
x[0,3] = 2000
x[0,4] = 19
x[0,5] = 72
x[0,6] = 1

prediction = model.predict(x)
print(float(prediction[0]))

json = {
    "cylinders": 8,
    "displacement": 300,
    "horsepower": 48,
    "weight": 3500,
    "acceleration": 20,
    "model year": 76,
    "origin": 1
}
r = requests.post('http://localhost:5000/api', json=json)
if r.status_code == 200:
    print(f'Works: {r.text}')
else:
    print(f'Failed: {r.text}')