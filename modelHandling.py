import pandas as pd
import joblib
import os
import sklearn
import lxml


def read_Data(csv_location):
    data = pd.read_csv(csv_location)
    return data


def read_Model(model_location):
    model = joblib.load(model_location)
    return model


def predict(data, model):
    prediction = model.predict(data.iloc[0:, :186])
    return prediction


def create_csv(data, prediction):
    output = pd.DataFrame({'id': [i for i in range(len(data.index))], "prediction": prediction})
    output.to_csv('output.csv', index=False)
    return output


def delete_csv():
    os.remove('output.csv')


def give_prediction(csv_location, model_location):
    data = read_Data(csv_location)
    model = read_Model(model_location)
    pred = predict(data, model)
    output = create_csv(data, pred)
    return output
