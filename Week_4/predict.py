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

# def prepare_features(year, month):

#     input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
#     output_file = f'./output/yellow_taxi_tripdata_{year:04d}-{month:02d}.parquet'
#     categorical = ['PULocationID', 'DOLocationID']

def prediction(year, month):
    # year = input_date['year']
    # month = input_date['month']
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    print(input_file)
    output_file = f'./output/yellow_taxi_tripdata_{year:04d}-{month:02d}.parquet'
    
    df = read_data(input_file)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    save_output(df, y_pred, output_file)

    return y_pred

def save_output(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
