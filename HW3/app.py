from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'wine_quality.pkl'
model = pickle.load(open(filename, 'rb'))
model = joblib.load(filename) 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    alcohol = request.form['alcohol']
    citric_acid = request.form['citric_acid']
    free_sulfur_dioxide = request.form['free_sulfur_dioxide']
    sulphates = request.form['sulphates']
    pH = request.form['pH']
    prediction = model.predict(np.array([[alcohol, citric_acid, free_sulfur_dioxide, sulphates, pH]]))
    #print(prediction)
    return render_template('index.html', predict=str(prediction))
if __name__ == '__main__':
    app.run(debug=True)