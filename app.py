from flask import Flask, redirect, render_template, request, url_for
import pickle
import numpy as np
import pandas as pd
import os

dataset_path = os.path.join(os.path.dirname(__file__), 'heart_failure_clinical_records_dataset.csv')
df = pd.read_csv(dataset_path)

filename = 'model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        cpk = int(request.form['CPK'])
        plate = float(request.form['platelets'])
        sc = float(request.form['SC'])
        ss = int(request.form['SS'])
        ef = int(request.form['EF'])
        time = int(request.form['time'])
        smoke = int(request.form['Smoking'])
        anae = int(request.form['anaemia'])
        pressure = int(request.form['bloodpressure'])
        dia = int(request.form['Diabetes'])
        sex = int(request.form['Gender'])

        data = np.array([[age, anae,cpk,dia,ef,pressure,plate,sc,ss,sex,smoke,time]])
        my_prediction = classifier.predict(data)[0]

        if my_prediction == 1:
            prediction_text = "Contact a nearby Heart Doctor Immediately!"
        else:
            prediction_text = "You are safe!"
        
        return render_template('result.html', prediction_text=prediction_text)

    # If the method is not POST, redirect to the home page
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
