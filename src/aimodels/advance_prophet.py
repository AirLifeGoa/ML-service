from datetime import datetime,timedelta
import pandas as pd
import logging
from prophet import Prophet
from prophet.plot import plot_plotly
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from math import sqrt
import numpy as np
import warnings
from util.best_model import BestModel
from util.upload_data import upload_data_to_db
from mlflow.tracking import MlflowClient
from matplotlib import pyplot as plt
import mlflow
warnings.filterwarnings('ignore')

from worker.workflow import Workflow
from aimodels.base_model import BaseModel
from data_factory.data_factory import ModelDataFactory

# import prepare_data as pr

# import test_perform as tp
ws =['auto',False]

class AdvanceProphet(BaseModel):

    def __init__(self, config, station_id, station_name, workflow: Workflow):
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.end_run()
        self.config = config
        self.train_data = None
        self.test_data =  None
        self.data_columns = []
        self.station_id =station_id
        self.datasource_name = station_name
        self.data_factory =  workflow.data_processor

        self.parse_config()
    
    def parse_config(self):
        self.interval_width = self.config["interval_width"]
        self.n_pred = self.config["days_ahead"]
        self.sps  = self.config["sps"]
        self.cps  = self.config["cps"]
        self.sm =  self.config["sm"]
        self.ws  = self.config["ws"]
        self.plot_range = self.config["plot_range"]
        self.plot_pred_test = self.config["plot_pred_test"]

    def train(self, train_data, test_data):
        self.train_data = train_data
        self.test_data  = test_data

        try: 
            exp_id = mlflow.create_experiment(f"{self.datasource_name} - Forecast Models")
        except:
            exp_id = mlflow.get_experiment_by_name(f"{self.datasource_name} - Forecast Models").experiment_id

        print(exp_id)
        # print(exp_id.__dict__["_experiment_id"])
        # for station_id in range(1,20):
        #Num prediction
        # n_pred = 240
        
        # self.data_columns = test_data.columns
        # print(self.data_columns)
        #Changepoints feature and holiday features build
        changepoints, holidays = self.data_factory.holiday_changepoints_build()
    
        #Run model and produce results
        mlflow.start_run(experiment_id=exp_id, run_name=f"Prophet - {self.data_factory.output} - { datetime.now().replace(second=0, microsecond=0)}")

        error_metric = self.perform_prophet(self.station_id,train_data = train_data,
                        test_data= test_data,
                        n_pred= self.n_pred,
                        changepoints= changepoints,
                        holidays= holidays,
                        ws= self.ws,
                        station_no= self.station_id,
                        save_fig= True,
                        plot_pred_test=self.plot_pred_test, 
                        plot_range= self.plot_range)

        full_data = pd.concat([self.train_data, self.test_data], axis=0)
        self.train_and_save_full_model(full_data, changepoints= changepoints,
                        holidays= holidays,
                        sps=10.0,
                        cps=0.05,
                        ws = 'auto',
                        sm='additive',
                        add_national_holiday=False)
        mlflow.end_run()

        return error_metric

    def run_prophet(self,station_id,train_data, n_pred,
                        changepoints=None,
                        holidays=None,
                        sps=10.0,
                        cps=0.05,
                        ws = 'auto',
                        sm='additive',
                        add_national_holiday=False):
        
        mlflow.set_tag("With Changepoints","True")
        mlflow.set_tag("With Holidays","False")
       
        mlflow.log_param("seasonality_prior_scale",str(sps))
        mlflow.log_param("changepoint_prior_scale",str(cps))
        mlflow.log_param("interval_width", "0.95")
        mlflow.log_param("weekly_seasonality", ws)
        mlflow.log_param("seasonality_mode", sm)

        model = Prophet(interval_width=self.interval_width,
                        changepoints=changepoints,
                        holidays=holidays,
                        seasonality_prior_scale=sps,
                        changepoint_prior_scale=cps,
                        weekly_seasonality= ws,
                        seasonality_mode=sm)

        if add_national_holiday == True:
            model.add_country_holidays('IN')
        
        model.fit(train_data)
        # mlflow.prophet.log_model(model, artifact_path="model", registered_model_name=f"{self.datasource_name}_{self.data_factory.output}_prophet")
        print(self.test_data)
        print("number of pred days",(self.test_data['ds'].max() - self.test_data['ds'].min()).days)
        future_dates = model.make_future_dataframe(periods=(self.test_data['ds'].max() - self.test_data['ds'].min()).days, include_history=False)
        prediction = model.predict(future_dates)
        fitting_graph = plot_plotly(model, prediction)
        
        model.plot_components(prediction).savefig('./save_plots/Components_of_model.png')
        mlflow.log_artifact("./save_plots/prophet_models/Components_of_model.png",artifact_path="plots")

        model.plot(prediction).savefig("./save_plots/predictions_of_model.png")
        mlflow.log_artifact("./save_plots/prophet_models/predictions_of_model.png",artifact_path="plots")

        return prediction, fitting_graph
    

    def perform_prophet(self,station_id,train_data, test_data,
                    n_pred,
                    changepoints=None,
                    holidays=None,
                    sps=10.0,
                    cps=0.25,
                    sm='additive',
                    ws ='auto',
                    plot_range=False,
                    plot_pred_test = False,
                    set_yrange=False,
                    station_no=None,
                    save_fig=False):
        # try:

            # Model with lockdown as changepoints.
            prediction_with_lockdown, fit_holiday = self.run_prophet(station_id,train_data, n_pred=n_pred, changepoints=changepoints,
                                                                ws=ws)
            
            pred = prediction_with_lockdown.tail(self.n_pred)
            pred_org = pred[['ds', 'yhat']].set_index('ds')
            # print("jhgvtf ",pred_org)
            print(test_data[["y"]])
            upload_data = pred_org.copy()
            upload_data = upload_data.rename(columns={'yhat': 'PM10'})
            # upload_data_to_db(upload_data, "prophet",station_id)

            
            mod_pred = pred_org.copy()
            dummy_test = test_data.copy()
            dummy_test = dummy_test.set_index('ds')
            # print(mod_pred, dummy_test)
            mod_pred['y'] = dummy_test[["y"]]
            # print("after adding y",mod_pred)
            mod_pred = mod_pred.dropna()
            
            # print(len(mod_pred))
            
            error_rmse = sqrt(mean_squared_error(mod_pred.yhat, mod_pred.y))
            error_mae = mean_absolute_error(mod_pred.yhat, mod_pred.y)
            
            mlflow.log_metric("RMSE Score", error_rmse)
            mlflow.log_metric("MAE Score",error_mae)
            print(f'Model with lockdown as change points --->  RMSE: {error_rmse:.2f}, MAE: {error_mae:.2f}')
            print(plot_pred_test, plot_range)

            plt.plot(mod_pred.drop(['y'], axis=1), color='#ebd500', label='Predicted (lockdown as changepoints)')
            plt.savefig("./save_plots/prophet_models/predicitions.png")
            mlflow.log_artifact("./save_plots/prophet_models/predicitions.png",artifact_path="plots")
            if plot_pred_test == True and plot_range == False:
                plt.plot(mod_pred.drop(['y'], axis=1), color='#ebd500', label='Predicted (lockdown as changepoints)')
                plt.savefig("./save_plots/prophet_models/predicitions.png")
                mlflow.log_artifact("./save_plots/prophet_models/predicitions.png",artifact_path="plots")
            elif plot_pred_test == False and plot_range == True:
                pred = prediction_with_lockdown.tail(n_pred).set_index('ds')
                plt.fill_between(pred.index, pred['yhat_lower'], pred['yhat_upper'], edgecolor='#ebd500',
                                facecolor='#ebd500', alpha=0.3, label='Predicted (lockdown as changepoints)')
            elif plot_pred_test == True and plot_range == True:
                print(f'Wrong choice of plot settings!!')
            else:
                plt.plot(pred_org, color='#ebd500', label='Predicted (lockdown as changepoints)')
            
            return {"RMSE Error: " :error_rmse ,"MAE Error: ": error_mae}
        # except:
        #     print('Error occured while building model!!!')
    
    def train_and_save_full_model(self,full_data, 
                        changepoints=None,
                        holidays=None,
                        sps=10.0,
                        cps=0.05,
                        ws = 'auto',
                        sm='additive',
                        add_national_holiday=False):
        
        print("fullll datattattatt",full_data)
        model = Prophet(interval_width=self.interval_width,
                        changepoints=changepoints,
                        holidays=holidays,
                        seasonality_prior_scale=sps,
                        changepoint_prior_scale=cps,
                        weekly_seasonality= ws,
                        seasonality_mode=sm)

        if add_national_holiday == True:
            model.add_country_holidays('IN')

        model.fit(full_data)
        model_info = mlflow.prophet.log_model(model, artifact_path="model", registered_model_name=f"{self.datasource_name}_{self.data_factory.output}_prophet")
        print(model_info)

    def inference(self, model: Prophet, test_inputs: list = None, days_ahead = None, start_date = None ,  end_date = None, from_train = False):
        
        if start_date == None:
            start_date = datetime.now().date()
            print("Start date is not provided taking default value")
        
        if end_date == None:
            end_date = start_date + timedelta(days_ahead)

        print("days_ahead",days_ahead)
        future_dates = model.make_future_dataframe(periods=days_ahead, include_history=False)
        print(future_dates)
        predictions = model.predict(future_dates)

        pred_org = predictions[['ds', 'yhat']].set_index('ds')
        pred_org = pred_org.rename(columns={'yhat': self.data_factory.output})  

        print(pred_org, len(pred_org))
        pred_org = pred_org[pred_org.index >= pd.to_datetime(start_date)]      
       
        # print(pred_org.head)
        return pred_org
    
    def save_data(self, data, datasource_id= None):
        if datasource_id == None:
            datasource_id = self.station_id
        upload_data_to_db(data, "hybridlstm", datasource_id)


if __name__=="__main__":
    obj = AdvanceProphet()
    obj.run_model()