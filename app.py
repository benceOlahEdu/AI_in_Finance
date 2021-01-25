import requests
from flask import Flask, request, jsonify
app = Flask(__name__)
import pandas as pd
import quandl
import math
import random
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

@app.route('/getstockdata/')
def getStockData():
    stock = request.args.get('stock', default=None, type=None)
    quandl.ApiConfig.api_key = "E9nzkgMZnt67Hdep7DJe"
    allData = quandl.get('WIKI/'+stock)

    print("All data from the random stock:\n" + allData)

    dataLength = 251
    allDataLength = len(allData)
    firstDataElem = math.floor(random.random()*(allDataLength-dataLength))
    mlData = allData[0:firstDataElem+dataLength]

    def FormatForModel(dataArray):
        dataArray = dataArray[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
        dataArray['HL_PCT'] = (dataArray['Adj. High'] - dataArray['Adj. Close']) / dataArray['Adj. Close'] * 100.0 #not used column
        dataArray['PCT_change'] = (dataArray['Adj. Close'] - dataArray['Adj. Open']) / dataArray['Adj. Open'] * 100.0 #not used column

        dataArray = dataArray[['Adj. Close', 'HL_PCT', 'PCT_change','Adj. Volume']]

        dataArray.fillna(-99999, inplace=True) #fill NA/NaN values with -99999

        return dataArray

    mlData = FormatForModel(mlData)

    print("Formatted data for stock:\n" + mlData)

    forecast_col = 'Adj. Close'
    forecast_out = int(math.ceil(0.12*dataLength))

    mlData['label'] = mlData[forecast_col].shift(-forecast_out) #label is the shifted back future Adj. Close column
    mlData.dropna(inplace=True) #Drop the NA/NaN values

    X = np.array(mlData.drop(['label'],1)) #INPUT without label
    X = preprocessing.scale(X) #Scale for machine learning

    X_data = X[-dataLength:] #Last "datalength" elements for PREDICTION
    X = X[:-dataLength] #First elements without the last "datalength" elements for TRAINING

    data = mlData[-dataLength:] #Last "datalength" elements with label for PREDICTION
    mlData = mlData[:-dataLength] #First elements" with label until the last "datalength" elements for TRAINING
    y = np.array(mlData['label']) #Only label for OUTPUT

    #Train for "first" elements
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3) #0.7 for train, 0.3 for test

    #Set the model type
    clf = LinearRegression()

    #Fit model for first datalength elements
    clf.fit(X_train, y_train)

    #Test the model with test elements
    accuracy = clf.score(X_test, y_test)

    print("Accuracy: " + accuracy)

    #Predict for the last "datalength" elements
    prediction = clf.predict(X_data)

    #Only EOD (Current value) and predicted value (for future in forecast_out)
    data = data[['Adj. Close']]
    data = data.rename(columns={'Adj. Close':'EOD'})
    data['prediction'] = prediction[:]

    #Send it out in JSON for javascript which check the difference between the EOD and prediction with a multiplier, and
    #and check it whether it bigger or smaller from an exect value and decide for buying or selling
    data = data.to_json(orient='table')
    return jsonify(data)
