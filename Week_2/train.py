import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output2",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    
    # start mlflow
    with mlflow.start_run():
        
        # use autolog
        mlflow.sklearn.autolog()

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    
    # Setting the tracking_uri for mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Create a new experiment for green_taxi
    mlflow.set_experiment("green_taxi_experiment")
    
    run_train()
