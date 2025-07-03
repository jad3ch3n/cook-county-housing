
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def remove_outliers(data, column, lower=None, upper=None):
    '''Remove outliers from a column based on lower and upper thresholds.'''
    if lower is not None:
        data = data[data[column] >= lower]
    if upper is not None:
        data = data[data[column] <= upper]
    return data

def log_transform(data, column):
    '''Add a log-transformed version of a column.'''
    new_column = f"Log {column}"
    data[new_column] = np.log(data[column])
    return data

def add_total_bedrooms(data):
    '''Extract number of bedrooms from the Description column using regex.'''
    data = data.copy()
    data['Bedrooms'] = data['Description'].str.findall(r' (\d+) rooms,').str[0].fillna(0).astype(int)
    return data

def train_val_split(data, test_size=0.2, random_state=1337):
    '''Split the data into training and validation sets.'''
    return train_test_split(data, test_size=test_size, random_state=random_state)
