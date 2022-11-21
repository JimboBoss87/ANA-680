from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'water_quality.pkl'
model = pickle.load(open(filename, 'rb')) 
model = joblib.load(filename) 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    ph = request.form['ph']
    Hardness  = request.form['Hardness']
    Solids = request.form['Solids']
    Chloramines = request.form['Chloramines']
    Sulfate  = request.form['Sulfate']
    Conductivity = request.form['Conductivity']
    Organic_carbon = request.form['Organic_carbon']
    Trihalomethanes = request.form['Trihalomethanes']
    Turbidity = request.form['Turbidity']
    prediction = model.predict(np.array([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity,Organic_carbon, Trihalomethanes, Turbidity]]))
    #print(prediction)
    return render_template('index.html', predict=str(prediction))
if __name__ == '__main__':
    app.run(debug=True)