from datetime import datetime,timedelta
import pandas as pd
import torch
import os
import joblib
from torch import nn
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
import mlflow
warnings.filterwarnings('ignore')

from worker.workflow import Workflow
from aimodels.base_model import BaseModel
from data_factory.data_factory import ModelDataFactory

#TODO make lstm as custom instead of hardcoding the layers.
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size= 300, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(hidden_layer_size, 100)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(100,output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        lstm_out = self.dropout1(lstm_out)
        predictions = self.linear2(self.dropout2(self.linear1(lstm_out.view(len(input_seq), -1))))
        return predictions[-1]

class CustomLSTM(BaseModel):

    def __init__(self, config, datasourceId, datasource_name, workflow: Workflow):
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.end_run()
        self.config = config
        self.train_data = None
        self.test_data =  None
        self.data_columns = []
        self.datasourceId =datasourceId
        self.datasource_name = datasource_name
        self.data_factory =  workflow.data_processor
        self.window_size = workflow.data_processor.window_size

        self.parse_config()
    
    def parse_config(self):
        self.epochs = self.config["epochs"]
        self.n_pred = self.config["days_ahead"]
        self.optimizer_name  = self.config["optimizer"]
        self.loss_function  = self.config["loss_function"]
        self.lr =  self.config["lr"]
        self.dropout  = self.config["dropout"]




    def get_loss(self):
        if self.loss_function == "MSE":
            return nn.MSELoss()
        else:
            raise Exception("Unable to find loss Function. Please specify correct loss function")
    
    def get_optimizer(self):
        if self.optimizer_name == "Adam":
            return torch.optim.Adam

    def build_model(self):
        model = LSTM()
        loss_function = self.get_loss()
        optimizer = self.get_optimizer()(model.parameters(), lr=self.lr)

        return model, loss_function, optimizer

    def train(self, train_data, test_data):
        self.train_data = train_data
        self.test_data  = test_data

        self.model, self.loss,self.optimizer = self.build_model()
        #training starts here

        self.model.train()
        for i in range(self.epochs):
            for seq, labels in self.train_data:
                self.optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                torch.zeros(1, 1, self.model.hidden_layer_size))

                y_pred = self.model(seq)

                single_loss = self.loss(y_pred, labels)
                single_loss.backward()
                self.optimizer.step()

            if i%25 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

        last_training_point = np.array(self.train_data[-1][0].tolist() + self.train_data[-1][1].tolist()).reshape(-1)
        
        last_training_point = list(last_training_point)
        prediction_data = self.inference(self.model,last_training_point[1:], len(test_data))
        error_metric = self.evaluate(self.test_data, prediction_data)

        
        self.save_plots(prediction_data, test_data)
        self.train_and_save_full_model()
        print(error_metric)
        return error_metric

    def train_and_save_full_model(self):
        #save model using mlflow 
        #create model here then pass it save model function.
        model, loss_function, optimizer = self.build_model()
        print(self.data_factory.full_data[0])
        for i in range(self.epochs):
            for seq, labels in self.data_factory.full_data:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            if i%25 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
        self.save_model(model)

    def evaluate(self,actual_data, prediction_data):
        """
        This funtion evaluates the performance of the model

        Args:
            prediction_data (_type_): last row in training data
            actual_data (_type_):  true data
        """
        # measured_data = actual_data[self.data_factory.output].to_numpy()
        mae_error = mean_absolute_error(prediction_data.squeeze(),actual_data)
        rmse_error = np.sqrt(mean_squared_error(prediction_data.squeeze(),actual_data))
        print("MAE: {}, RMSE: {}".format(mae_error,rmse_error))
        # print(self.config)
        return {"RMSE": rmse_error, "MAE": mae_error}
    
    def save_plots(self,prediction_data,test_data):
        print(test_data.head())
        plt.plot(test_data)
        plt.plot(test_data.index, prediction_data)
        print(self.datasource_name)
        plt.savefig(f"./save_plots/{self.datasource_name}/lstm/predicitions.png")
        plt.show()

    def save_model(self, model):
        try: 
            exp_id = mlflow.create_experiment(f"{self.datasource_name} - Forecast Models")
        except:
            exp_id = mlflow.get_experiment_by_name(f"{self.datasource_name} - Forecast Models").experiment_id

        print(exp_id)
        #Run model and produce results 
        mlflow.start_run(experiment_id=exp_id, run_name=f"LSTM - {self.data_factory.output} - { datetime.now().replace(second=0, microsecond=0)}")
        # TODO write model to save it in mlflow with sclar, model and params & plots.
        
        mlflow.pytorch.log_model(
            model,
            artifact_path= "forecast_model",
            registered_model_name=f"{self.datasource_name}_{self.data_factory.output}_lstm"
        )

        joblib.dump(self.data_factory.scalar, "scaler.pkl")
        mlflow.log_artifact("scaler.pkl")
        os.remove("scaler.pkl")

        mlflow.log_param("model_type", "LSTM")
        # mlflow.log_param("input_features", self.data_factory.inputs)
        mlflow.log_param('days_ahead', self.n_pred)
        mlflow.log_param('epochs', self.epochs)
        mlflow.log_param('optimizer', self.optimizer_name) 
        mlflow.log_param('loss_function', self.loss_function)
        mlflow.log_param('lr', self.lr)
        mlflow.log_param( 'hidden_layers', self.config['hidden_layers'])
        mlflow.log_param( 'dropout', self.dropout)
        mlflow.log_param("features_to_forecast", self.data_factory.output)
        mlflow.log_param("model_trained", True)
        # mlflow.log_artifact(f"./save_plots/{self.datasource_name}/lstm/predicitions.png",artifact_path="plots")
        mlflow.end_run()
   
    
    def inference(self, model: LSTM, test_inputs: list, days_ahead = 50, start_date = datetime.now().date(), end_date = None,):
        """
        This function do predictions for given number of days ahead

        Args:
            model (LSTM): lstm model
            test_inputs (list): this is the last training datapoint 
            days_ahed (int): Number of days forecasts needs to be done
        """
        print(start_date)
        print(days_ahead)
        model.eval()

        for i in range(days_ahead):
            seq = torch.FloatTensor(test_inputs[-self.window_size:])
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))
                test_inputs.append(model(seq).item())

        actual_predictions = self.data_factory.scalar.inverse_transform(np.array(test_inputs[self.window_size:] ).reshape(1, -1))[0]

        days = pd.date_range(start_date, start_date + timedelta(days_ahead-1), freq='D')
        print(len(actual_predictions), days_ahead, len(days))

        preds = pd.DataFrame({'date': days, self.data_factory.output: actual_predictions})
        preds = preds.set_index('date')
        print(preds.head())
        return preds
    

    def save_data(self, data, datasource_id= None):
        if datasource_id == None:
            datasource_id = self.datasourceId
        upload_data_to_db(data, "lstm", datasource_id)
        


if __name__=="__main__":
    obj = CustomLSTM()