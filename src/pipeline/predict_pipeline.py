import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object, convert_to_price

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            return model.predict(data_scaled)
        except Exception as e:
            raise CustomException(e, sys) 
        

#  year = int(request.form.get('year')),
#             car_make = request.form.get('car_make'),
#             car_model = request.form.get('car_model'),
#             engine_size = float(request.form.get('engine_size')),
#             horsepower = int(request.form.get('horsepower')),
#             torque = int(request.form.get('torque')),
#             acceleration_time 
class CustomData:
    def __init__(self, 
                 year : int,
                 car_make: str,
                 car_model,
                 engine_size: float,
                 horsepower: float,
                 torque:float,
                 acceleration_time:float
                 ):
        
        self.year = year
        self.car_make = car_make
        self.car_model = car_model
        self.engine_size = engine_size
        self.horsepower = horsepower
        self.torque = torque
        self.acceleration_time = acceleration_time

    def get_acceleration_category(self):
        if self.acceleration_time < 5.0:
            return "Fast"
        elif 5.0 <= self.acceleration_time < 7.0:
            return "Moderate"
        else:
            return "Slow"
    def categorize_data(self, value, bins, labels):
        return pd.cut([value], bins=bins, labels=labels)[0]
    def get_engine_displacement_category(self):
        bins = [0, 2.0, 3.0, 4.0, float('inf')]
        labels = ['Small', 'Medium', 'Large', 'Very Large']
        return self.categorize_data(self.engine_size, bins, labels)
#'Year', 'Engine Size (L)', 'Horsepower', 'Torque (lb-ft)',\n       '0-60 MPH Time (seconds)'],
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "Year": [self.year],
                "Car Make": [self.car_make],
                "Car Model": [self.car_model],
                "Engine Size (L)": [self.engine_size],
                "Horsepower": [self.horsepower],
                "Torque (lb-ft)": [self.torque],
                "0-60 MPH Time (seconds)": [self.acceleration_time],
                "Engine Displacement Category": [self.get_engine_displacement_category()],
                "Acceleration Category": [self.get_acceleration_category()]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)