import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('fifa_players (1).csv')
df.drop(['name', 'full_name', 'birth_date', 'national_team', 'national_rating', 'national_team_position',
         'national_jersey_number'], axis=1, inplace=True)


def fill_nan_by_mean(column):
    df[column].fillna(df[column].mean(), inplace=True)


def log_column(column):
    new_column = column + '_log'
    df[new_column] = np.log1p(df[column])
    df.drop(column, axis=1, inplace=True)


def preprocess_height():
