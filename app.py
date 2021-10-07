from flask import Flask,render_template,request
import joblib
import os
import numpy as np
from flask import  jsonify
import pickle

from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('sample.html')

@app.route("/result",methods =['POST','GET'])
def result():
    list_col = ['Country', 'shipment_Mode', 'item_description', 'Brand', 'Unit_measure',
       'line_Item_Quantity', 'manufacturing_Site', 'first_Line_Designation',
       'weight_in_kg', 'Insurance_amount']

    Country = request.form['Country']
    shipment_Mode = request.form['shipment_Mode']
    item_description = request.form['item_description']
    Brand = request.form['Brand']
    Unit_measure = float(request.form['Unit_measure'])
    line_Item_Quantity = float(request.form['line_Item_Quantity'])
    manufacturing_Site = request.form['manufacturing_Site']
    first_Line_Designation = int(request.form['first_Line_Designation'])
    weight_in_kg = float(request.form['weight_in_kg'])
    Insurance_amount = float(request.form['Insurance_amount'])








    # Normalising data

    Unit_measure = np.log10(Unit_measure )

    log_max = 13.337473144116748
    line_Item_Quantity = (line_Item_Quantity) ** (1 / log_max)

    weight_in_kg = np.log1p(weight_in_kg)

    log_max = 8.950071111454612
    Insurance_amount = (Insurance_amount) ** (1 / log_max)

    # Lets put all in list

    inputs = np.array([Country, shipment_Mode, item_description, Brand, Unit_measure,
                       line_Item_Quantity, manufacturing_Site, first_Line_Designation,
                       weight_in_kg, Insurance_amount]).reshape(1,-1)



    # Applying standard scalar

    min_max = joblib.load(r'C:\Users\Hp\Desktop\Intership\Logistics\Models\min.sav')

    inputs_std = min_max.transform(inputs)

    # Applying prediction model

    model = joblib.load(r'C:\Users\Hp\Desktop\Intership\Logistics\Models\model.sav')

    prediction = model.predict(inputs_std)
    prediction = np.expm1(prediction)
    prediction = prediction.tolist()

    return render_template('predict.html',cost = prediction)
    #return jsonify({'prediction' : prediction})







if __name__ == '__main__':
    app.run(debug=True,port=7891)