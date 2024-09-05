from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

import pickle
import os
import csv

def create_preproc_pipeline() -> Pipeline:
    """
    Create a pipeline for preprocessing data, with parrallel processing for 
    numeric and categorical features
    return: a pipeline object
    """
    num_preproc_pipe = Pipeline([("Imputer",SimpleImputer())
                                ,("Scaling",StandardScaler())
                                ])

    cat_preproc_pipe = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                                 ("Encode",OneHotEncoder(drop="first", handle_unknown="ignore" ))
                                ])
    preproc_pipe = ColumnTransformer([("NumPreproc",num_preproc_pipe,make_column_selector(dtype_include="number")),
                                      ("CatPreproc",cat_preproc_pipe,make_column_selector(dtype_include="object"))
                                    ])
    return preproc_pipe

    
    
def create_model_pipeline() -> Pipeline:
    """
    Create  training a model
    return: a pipeline or model object
    """
    return VotingRegressor([("rand",RandomForestRegressor(min_samples_leaf=5)),
                            ("lin",LinearRegression()),
                            ("knn",KNeighborsRegressor())]
                            )


def save_model(model: Pipeline, filename: str) -> None:
    """
    Save the model to the models folder
    """
    if not os.path.exists("models") :
        os.mkdir("models")
    
    with open(f"./models/{filename}", "wb") as f:
        pickle.dump(model, f)
    

def save_metrics(metrics: dict, filename: str):
    """
    Save the metrics to the metrics folder
    """
    if not os.path.exists("metrics") :
        os.mkdir("metrics")

    with open(f"./metrics/{filename}", "w", newline="") as f:
        w = csv.DictWriter(f, metrics.keys())
        w.writeheader()
        w.writerow(metrics)
    

def load_pipeline(filename: str) -> Pipeline:
    with open(f"./models/{filename}", "rb") as f:
        return pickle.load(f)

def load_metrics(filename: str) -> dict:
    with open(f"./metrics/{filename}","rb") as f:
        metrics = {}
        metrics = csv.reader(f)
        return metrics

if __name__ == "__main__":
    dummy_dict = {"hello":"world", "moi":"toi"}
    save_metrics(dummy_dict, "test.csv")