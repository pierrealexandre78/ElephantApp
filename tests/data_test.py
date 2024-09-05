import pytest

import src.data as data

import pandas as pd


def test_load_data():
    data.load_data()
    data = pd.read_csv('data/train.csv')
    # check the number of columns in the dataframe
    assert(data.shape[1] == 81)

def test_clean_data():
    df = data.load_data()
    cleaned_df = data.clean_data(df)
    #check if the NAN are dropped
    assert(cleaned_df.isna().sum() ==  0)
    #check if the duplicates are dropped
    assert(cleaned_df.duplicated().sum() == 0)
    
