# time series

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.plot import plot_plotly

from pmdarima.arima import auto_arima

import plotly.offline as py
py.init_notebook_mode()
plt.style.use('fivethirtyeight')


# load data
data_train_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

data_train_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

# reset index
data_train_a.reset_index(inplace=True)
data_train_a.rename(columns={'datetime': 'ds', 'cpu': 'y'}, inplace=True)

data_train_b.reset_index(inplace=True)
data_train_b.rename(columns={'datetime': 'ds', 'cpu': 'y'}, inplace=True)

#model
m = Prophet()
m.fit(data_train_a)

future = m.make_future_dataframe(periods=1)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#changing params
m1 = Prophet(weekly_seasonality=False)
m1.add_seasonality(name='hourly', period=60, fourier_order=5)
forecast = m1.fit(data_train_a).predict(future)
fig = m1.plot_components(forecast)

m2 = Prophet(changepoint_prior_scale=0.01).fit(data_train_a)
future = m2.make_future_dataframe(periods=300, freq='1min')
fcst = m2.predict(future)
fig = m2.plot(fcst)

m3 = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
m3.fit(data_train_a)
future = m3.make_future_dataframe(periods=300, freq='1min')
fcst = m3.predict(future)
fig = m3.plot_components(fcst)

# model b
mb = Prophet()
mb.fit(data_train_b)
mb = Prophet(changepoint_prior_scale=0.01).fit(data_train_b)
future = mb.make_future_dataframe(periods=300, freq='1min')
fcst = mb.predict(future)
fig = mb.plot(fcst)

mb2 = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True).fit(data_train_b)
future = mb2.make_future_dataframe(periods=300, freq='1min')
fcst = mb2.predict(future)
fig = mb2.plot(fcst)

## ARIMA MODELS
data_train_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

data_train_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

data_train_a.index = pd.to_datetime(data_train_a.index)
data_train_b.index = pd.to_datetime(data_train_b.index)

#model 1
stepwise_model = auto_arima(data_train_a, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())

stepwise_model.fit(data_train_a)
stepwise_model.fit(data_train_a).plot_diagnostics(figsize=(15, 12))
plt.show()

future_forecast = stepwise_model.predict(n_periods=60)
future_forecast = pd.DataFrame(future_forecast,index = data_test_a.index,columns=['Prediction'])
pd.concat([data_test_a,future_forecast],axis=1).plot()


stepwise_model2 = auto_arima(data_train_a, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model2.aic())

stepwise_model2.fit(data_train_b)

stepwise_model2.fit(data_train_b).plot_diagnostics(figsize=(15, 12))
plt.show()

future_forecast = stepwise_model2.predict(n_periods=60)
future_forecast = pd.DataFrame(future_forecast,index = data_test_b.index,columns=['Prediction'])
pd.concat([data_test_b,future_forecast],axis=1).plot()