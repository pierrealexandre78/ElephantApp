from model import create_preproc_pipeline, create_model_pipeline, save_metrics
from data import load_data, clean_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# TODO : Turn the print into logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

seed = 42

def train():
    """
    train the data on the housing price data set
    """

    logger.info("Training the model")
    # Load the data
    df = load_data()

    logger.info("Data loaded successfully")

    # Clean the data
    clean_df = clean_data(df)

    logger.info(clean_df.head())

    # Create the preprocessor and the model

    preproc =  create_preproc_pipeline()
    model = create_model_pipeline()

    # Split the data into train and test sets

    X = clean_df.drop(columns=['SalePrice'])
    y = clean_df['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    logger.info("Split the data into train and test sets")
    logger.info(X_train.shape)
    logger.info(X_test.shape)

    # Preprocess the data

    # Fit the model
    X_train_preproc = preproc.fit_transform(X_train)


    model.fit(X_train_preproc, y_train)
    
    # Evaluate the model
    X_test_preproc = preproc.transform(X_test)

    y_pred = model.predict(X_test_preproc)

    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    
    logger.info(f"Mean Absolute Error: {mae}")
    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"R2 Score: {r2}")
    logger.info("Model trained successfully")
    
    # Save the MAE, MSE and R2 score in the metrics folder

    metrics = {"MAE": mae,
               "MSE": mse,
               "R2": r2
               }
    
    save_metrics(metrics, filename=f"metrics.csv")

if __name__ == "__main__":
    train()