
### Question 1

```bash
# create a separte virtual environment
python -m venv mlenv

# activate a new virtual env
. mlenv/Scripts/activate

# install mlflow
pip install mlflow

mlflow --version
# mlflow, version 2.3.2
```

### Question 2

1. Create a directory for storing data:
```bash
mkdir -p data/2022
```

2. Create a .txt file with links to data for download:
```ny_data.txt
https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet
https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-02.parquet
https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-03.parquet
```
3. Download the data:
```bash
wget -i ny_data.txt -P data/2022
```
4. Preprocsses the files
```bash
python preprocess_data.py --raw_data_path ./data/2022 --dest_path ./output
```
5. The size of DictVectorizer is aroun 154 kB:
```bash
ls -lh output/dv.pkl
```

### Question 3
1. The original file has been changed as shown below:
```python
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
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_train(data_path: str):
    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ny-taxi-2022")
    mlflow.sklearn.autolog()
    run_train()
```
2. The max_depth parameter equals 10.

### Question 4

1. Create a new folder.
```bash
mkdir artifact
```
2. Launche the tracking server
```bash
mlflow uri --backend-store-ure sqlite:///mlflow.db --default-artifact-root ./artifacts
```
3. Install optuna library
```bash
pip install optuna
```
4. The objection function was modified as shown below:
```python
    def objective(trial):
        with mlflow.start_run():
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 50, 1),
                'max_depth': trial.suggest_int('max_depth', 1, 20, 1),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10, 1),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4, 1),
                'random_state': 42,
                'n_jobs': -1
            }

            mlflow.log_params(params)
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)
        return rmse
```
5. The best validation RMSE is 2.45

### Question 5

1. Added the following lines to the run_register_model function.
```python
# Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(experiment_ids=experiment.experiment_id,
                                  max_results=1,
                                  order_by=["metrics.test_rmse ASC"])[0]
                                  
    # Register the best model
    mlflow.register_model(model_uri=f"runs:/{best_run.info.run_id}/model",
                          name='ny-best-model')
```
2. The test rmse is 2.285

