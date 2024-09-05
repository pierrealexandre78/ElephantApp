import pandas as pd
import pickle
import os

def load_data() -> pd.DataFrame:
    """
    try to load the data from the data folder ( locally )
    if data does not exist , load it from the web and save it in the data folder
    """
    try:
        data = pd.read_csv('data/train.csv')
    except FileNotFoundError:
        data = pd.read_csv(os.environ.get('DATA_URL'))
        data.to_csv('data/train.csv', index=False)
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    try to clean the data by removing missing values and duplicates
    """
    # drop NAN and duplicates rows
    r_data = data.dropna(axis=1).drop_duplicates()

    # Set the index colomn as the dataframe index
    r_data.set_index(r_data['Id'], inplace=True)
    r_data.drop(columns=['Id'], axis=1, inplace=True)
    return r_data

if __name__ == "__main__":
    data = load_data()
    print(data.shape)
    df = data.copy()
    df.set_index(df['Id'], inplace=True)
    df.drop(columns=['Id'], inplace=True)
    print(df.head())