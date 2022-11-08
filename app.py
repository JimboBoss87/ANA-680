from flask import Flask, render_template, request
import numpy as np
import pickle
#import joblib
app = Flask(__name__)
filename = 'race_class.pkl'
model = pickle.load(open(filename, 'rb'))    # load the model
#model = joblib.load(filename)  # two ways to load the model, not using joblib here
#model = joblib.load('filename.pkl')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    math_score = request.form['math_score']
    reading_score = request.form['reading_score']
    writing_score = request.form['writing_score']
    prediction = model.predict(np.array([[math_score, reading_score, writing_score]]))
    #print(prediction)
    return render_template('index.html', predict=str(prediction))
if __name__ == '__main__':
    app.run(debug=True)
