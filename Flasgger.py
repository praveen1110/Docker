import flask
import pandas as pd
from flask import Flask,render_template,request,jsonify
import joblib
import flasgger
from flasgger import Swagger
import pickle
# model = joblib.load(r'/Users/praveen/PycharmProjects/Docker/rfc.sav')
pickle_in = open("classifier.pkl","rb")
model=pickle.load(pickle_in)
app = Flask(__name__)

Swagger(app)

@app.route('/')
def welcome():
    return "Welcome Docker Flask"

@app.route('/home',methods=['GET'])
def home():

    """Docker Deployement
    ---
    parameters:
        -   name: variance
            in: query
            type: number
            required: true
            
        -   name: skewness
            in: query
            type: number
            required: true
            
        -   name: curtosis
            in: query
            type: number
            required: true
            
        -   name: entropy
            in: query
            type: number
            required: true
    responses:
            200:   
                description: The output values

    """
    variance = request.args.get('variance')
    skewness= request.args.get('skewness')
    curtosis= request.args.get('curtosis')
    entropy= request.args.get('entropy')
    data= [[variance, skewness, curtosis, entropy]]
    output= model.predict(data)
    # response= [data[0],output]
    return str(output[0])

@app.route('/predict_file',methods=["POST"])
def predict_api():
    """
    Docker Deployement
    ---
    parameters:
        -   name: file
            in: formData
            type: file
            required: true

    responses:
            200:
                description: The output values
    """
    data=pd.read_csv(request.files.get('file'))
    # Y=data[['class']]
    responses = model.predict(data.iloc[:,1:])
    responses = str(list(responses))
    return responses



if __name__ =='__main__':
    app.run(debug=True)