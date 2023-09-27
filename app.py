import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.utils import convert_to_price
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            year = int(request.form.get('year')),
            car_make = request.form.get('car_make'),
            car_model = request.form.get('car_model'),
            engine_size = float(request.form.get('engine_size')),
            horsepower = int(request.form.get('horsepower')),
            torque = float(request.form.get('torque')),
            acceleration_time = float(request.form.get('acceleration_time')),

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=convert_to_price(results[0]))
    
if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)