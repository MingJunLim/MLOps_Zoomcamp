import predict

# year = 2022
# month = 4

my_date = {
    "year": 2022,
    "month": 4
}

features = predict.prepare_features(my_date)
pred = predict.predict(features)
# predicted_data = predict_2.prediction(year, month)
print(pred.mean())