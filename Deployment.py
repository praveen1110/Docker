import flask
import pandas as pd
from flask import Flask,render_template,request,jsonify
import joblib
import json

model = joblib.load(r'/Users/praveen/PycharmProjects/Docker/rfc.sav')
app= Flask(__name__)

@app.route('/home',methods=['GET'])
def home():
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
    data=pd.read_csv(request.files.get('file'))
    # Y=data[['class']]
    responses = model.predict(data.iloc[:,1:])
    responses = str(list(responses))
    return responses



if __name__ =='__main__':
    app.run(debug=True)