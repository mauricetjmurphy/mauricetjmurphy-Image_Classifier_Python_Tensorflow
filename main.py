from flask import Flask
from werkzeug.wrappers import Request, Response

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import numpy as np

import os
import io
import requests


app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello World'

if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple('localhost', 9000, app)

# Reading in the data from csv
df = pd.read_csv('Data/auto-mpg.csv', na_values=['NA', '?'])

# Filling the null values will the median value
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

# Take only the relevent columns and convert them to a numpy array (.values)
x = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']].values
y = df['mpg'].values

# Splitting the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

model = Sequential()
model.add(Dense(25, input_dim=x.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)

model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor], verbose=2, epochs=1000)

prediction = model.predict(x_test)
score = np.sqrt(metrics.mean_squared_error(prediction, y_test))
print(f'After the training the score is {score}\n')

if not os.getcwd == 'Models':
    os.chdir('Models')
os.getcwd()
model.save(os.path.join(os.getcwd(), 'mpg_model.h5'))

cols = [x for x in df.columns if x not in ('mpg', 'car name')]

print('{')
for i, name in enumerate(cols):
    print(f'{name}: (min:{df[name].min()}, max: {df[name].max()}){"," if i<(len(cols)-1) else ""}')
print('}')