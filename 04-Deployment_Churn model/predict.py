# Saving and loading the model
## Saving the model to pickle
## Loading the model from pickle
## Turning our notebook into a python script

# Load the model

import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask('churn') # create a flask app and call the app `churn`

# add a decorator
@app.route('/predict', methods=['POST']) #use `POST` for sending information about the customer rather than ping
def predict():
    customer = request.get_json()
    
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5
    
    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)        
    }
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696) #curl http://localhost:9696/ping
    
    