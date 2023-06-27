import pickle
import pandas as pd

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


def prepare_features(input_date):
    year = input_date['year']
    month = input_date['month']
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    print(input_file)
    
    df = read_data(input_file)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = df[categorical].to_dict(orient='records')

    return dicts


def predict(features):
    X_val = dv.transform(features)
    y_pred = model.predict(X_val)
    
    return y_pred.mean()
