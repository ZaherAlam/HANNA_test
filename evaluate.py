from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(true_vals, pred_vals, squared=False)
