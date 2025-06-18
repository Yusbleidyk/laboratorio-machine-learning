import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    data = data.drop(data.columns[0:3], axis=1)
    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = data[column].fillna(data[column].mode()[0])
            data[column] = data[column].astype("category").cat.codes
        else:
            data[column] = data[column].fillna(data[column].median())
    return data
