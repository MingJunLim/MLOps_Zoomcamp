import predict

year = 2022
month = 3

# my_date = {
#     "year": 2022,
#     "month": 3
# }

predicted_data = predict.prediction(year, month)
print(predicted_data.mean())