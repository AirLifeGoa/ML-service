from datetime import datetime,timedelta
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from math import sqrt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class BasicProphet:

    def __init__():
        pass

    def run_model(train_data, n_pred,
                changepoints=None,
                holidays=None,
                sps=10.0,
                cps=0.05,
                ws = 'auto',
                sm='additive',
                add_national_holiday=False):

        model = Prophet(interval_width=0.95,
                        changepoints=changepoints,
                        holidays=holidays,
                        seasonality_prior_scale=sps,
                        changepoint_prior_scale=cps,
                        weekly_seasonality= ws,
                        seasonality_mode=sm)
        
        if add_national_holiday == True:
            model.add_country_holidays('IN')
        
        model.fit(train_data)

        future_dates = model.make_future_dataframe(periods=n_pred)
        prediction = model.predict(future_dates)
        fitting_graph = plot_plotly(model, prediction)

        return prediction, fitting_graph
    
    

