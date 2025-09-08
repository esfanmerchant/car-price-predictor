# Flask, pandas, scikit learn, pickle-mixin

from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open('LinearRegressionModel.pkl','rb'))
app = Flask(__name__)
car = pd.read_csv('car_data.csv')

@app.route('/')
def index():
    companies = sorted(car['Brand'].unique())
    fuel_type = sorted(car['Fuel_Type'].unique())
    tranmission = sorted(car['Tranmission'].unique())
    year = sorted(car['Year'].unique(), reverse=True)
    owner_type = ['First','Second','Third','Fourth & Above']
    return render_template("index.html", brand=companies, fuel_type = fuel_type, tranmission=tranmission, owner_type = owner_type, year=year)

@app.route('/predict',methods=['POST'])
def predict():
    brand = request.form.get('brand')
    year = int(request.form.get('year'))
    seats = int(request.form.get('seats'))
    kilo_driven = float(request.form.get('kilo_driven'))
    engine = float(request.form.get('engine'))
    power = float(request.form.get('power'))
    mileage = float(request.form.get('mileage'))
    owner_type = request.form.get('owner_type')
    fuel_type = request.form.get('fuel_type')
    tranmission = request.form.get('tranmission')

    prediction = model.predict(pd.DataFrame([[seats,engine,power, mileage,brand,year,kilo_driven,fuel_type,tranmission,owner_type]],columns=['Seats','Engine_(CC)','Power_(bhp)','Mileage_(kmpl)','Brand','Year','Kilometers_Driven','Fuel_Type','Tranmission','Owner_Type']))

    return str(np.round(prediction[0],2))

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)