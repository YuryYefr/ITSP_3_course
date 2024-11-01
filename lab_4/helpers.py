import pandas as pd

def process_data():
    data = pd.read_csv('./data.csv')
    data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    data.dropna(inplace=True)
    return data
