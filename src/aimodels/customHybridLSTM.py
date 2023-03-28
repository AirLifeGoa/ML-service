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
from sklearn.preprocessing import StandardScaler
import mlflow
warnings.filterwarnings('ignore')

from worker.workflow import Workflow
from aimodels.base_model import BaseModel
from data_factory.data_factory import ModelDataFactory



class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20 # number of hidden states
        self.n_layers = 1 # number of LSTM layers (stacked)
    
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden*self.seq_len, 1)
        
    
    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state, cell_state)
    
    
    def forward(self, x):        
        batch_size, seq_len, _ = x.size()
        
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.l_linear(x)
    


class CustomHybridLSTM(BaseModel):

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
        self.window_size = None

        self.parse_config()
    
    def parse_config(self):
        self.epochs = self.config["epochs"]
        self.n_pred = self.config["days_ahead"]
        self.optimizer_name  = self.config["optimizer"]
        self.loss_function  = self.config["loss_function"]
        self.lr =  self.config["lr"]
        self.dropout  = self.config["dropout"]
        self.weight_decay = self.config["weight_decay"]
        self.batch_size = self.config["batch_size"]
        self.window_size = self.config["window_size"]

    def get_loss(self):
        if self.loss_function == "MSE":
            return nn.MSELoss()
        else:
            raise Exception("Unable to find loss Function. Please specify correct loss function")
    
    def get_optimizer(self):
        if self.optimizer_name == "Adam":
            return torch.optim.Adam

    def build_model(self, n_features=32):
        model = MV_LSTM(n_features=n_features , seq_length = self.window_size)
        loss_function = self.get_loss()
        optimizer = self.get_optimizer()(model.parameters(), lr=self.lr, weight_decay = self.weight_decay)

        return model, loss_function, optimizer
    
    def create_lstm_data(self, prediction, train_data, test_data):
        #Creating a new dataset consisting of train and test 
        concat = train_data.copy()
        concat_y = test_data.copy()
        concat_y.set_index('ds',inplace = True)
        concat.set_index('ds',inplace = True)
        print("Only train data: {}".format(concat.shape))
        concat = pd.concat([concat,concat_y])
        print("Both train and test data: {}".format(concat.shape))

        #New dataset to feed to LSTM
        #Removing all the extra predictions that are not in measured data(concat)

        # print(concat)
        prediction_ = prediction[prediction.ds.isin(concat.index)]

        # print("after removing uneccesary stuff: ", prediction_.shape, prediction.shape)
        y_hat = prediction_.yhat.values
        # y_error = np.abs(prediction_.yhat.values - concat.y.values)
        prediction_.drop('yhat',axis = 1,inplace =True)
        # prediction_['y_error'] = y_error
        prediction_['y_hat'] = y_hat
        prediction_.set_index('ds',inplace = True)

        split_index = int(len(prediction_)*0.85)
        train_end = prediction_.index[split_index]

        lstm_train = prediction_[:train_end]
        lstm_test = prediction_[train_end+timedelta(days=1):]

        print(lstm_train.shape)
        print(lstm_test.shape)

        return lstm_train, lstm_test
    
    def scale_lstm_data(self,lstm_train, lstm_test):
        self.std = StandardScaler()
        lstm_train_transformed = self.std.fit_transform(lstm_train)
        lstm_test_transformed = self.std.transform(lstm_test)

        return lstm_train_transformed, lstm_test_transformed

    
    # split a multivariate sequence into samples
    def split_sequences(self, sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    def train_prophet(self, train_data, forecast_period = 250):
        changepoints, holidays = self.data_factory.holiday_changepoints_build()
        #Model running
        model = Prophet(interval_width=0.95,
                            changepoints=changepoints,
                            holidays=holidays,
                            seasonality_prior_scale= 10.0,
                            changepoint_prior_scale= 0.05,
                            weekly_seasonality= "auto",
                            seasonality_mode= "additive")

        #model.add_country_holidays('IN')
        model.fit(train_data)
        future_dates = model.make_future_dataframe(periods= forecast_period)
        prediction = model.predict(future_dates)
        return model, prediction
        

    def train_lstm(self, train_data_x, train_data_y):
        model, loss_func, optimizer = self.build_model()
        #training starts here

        model.train()
        for i in range(1, self.epochs+1):
            for b in range(0,len(train_data_x), self.batch_size):
                inpt = train_data_x[b:b+self.batch_size,:,:]
                target = train_data_y[b:b+self.batch_size] 
                
                x_batch = torch.tensor(inpt,dtype=torch.float32)    
                y_batch = torch.tensor(target,dtype=torch.float32)
    
                model.init_hidden(x_batch.size(0))
                output = model(x_batch) 
                loss = loss_func(output.view(-1), y_batch)  
                loss.backward()
                optimizer.step()        
                optimizer.zero_grad()

            if i%25 == 1:
                print(f'epoch: {i:3} loss: {loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {loss.item():10.10f}')

        return  model

    def rescale_data(self, preds, lstm_test_transformed):
        pred_numpy = np.array(preds)
        x =  lstm_test_transformed[:62,:-1]
        y = pred_numpy.reshape((-1,1))
        z = np.append(x,y,axis =1)
        z_transformed = self.std.inverse_transform(z)

    def train(self, train_data, test_data):

        # self.load_model()
        # exit()

        self.train_data = train_data
        self.test_data  = test_data

        # print(train_data, test_data)

        forecast_period =  max(pd.to_datetime(test_data['ds'])) - max(pd.to_datetime(train_data['ds'])) 

        _, prophet_predictions = self.train_prophet(train_data=train_data, forecast_period=forecast_period.days)
        print(prophet_predictions.shape, "############")

        lstm_train_data, lstm_test_data = self.create_lstm_data(prophet_predictions, train_data, test_data)
        print("training data size: ", lstm_train_data.shape,"testing data size: ", lstm_test_data.shape)

        lstm_train_transformed, lstm_test_transformed = self.scale_lstm_data(lstm_train_data, lstm_test_data)
        print(lstm_train_transformed.shape)
        lstm_final_train_x, lstm_final_train_y = self.split_sequences(lstm_train_transformed, self.window_size)
        lstm_final_test =  self.split_sequences(lstm_test_transformed, self.window_size)
        print("testing dataset size :",lstm_final_test[0].shape)

        self.model = self.train_lstm(lstm_final_train_x, lstm_final_train_y)
        self.lstm_test_transformed = lstm_test_transformed 
        
        # print(prophet_predictions.ds.values[-1].astype('datetime64[s]').astype('O'),type(prophet_predictions.ds.values[-1].astype('datetime64[s]').astype('O')))
        rescaled_prediction_data = self.inference(self.model, lstm_final_test, len(lstm_final_test[0]), start_date = prophet_predictions.ds.values[-1].astype('datetime64[s]').astype('O'),from_train=True)
        print("shape before inverse transform ",lstm_final_test[1].reshape(-1, 1).shape, rescaled_prediction_data.shape)

        # dummy_data = lstm_test_transformed.copy()[7:,:-1]
        # preds_numpy  = np.array(prediction_data).reshape((-1,1))
        
        # z = np.append(dummy_data,preds_numpy,axis =1)
        # rescaled_prediction_data = self.std.inverse_transform(z)[:,-1]    

        error_metric = self.evaluate(test_data[7:]['y'].values, rescaled_prediction_data[self.data_factory.output].values)

        # print(prediction_data.shape, lstm_final_test[1].shape, lstm_test_data.shape)
        self.save_plots(rescaled_prediction_data[self.data_factory.output].values, test_data[7:]['y'].values, lstm_test_data[:-7])

        self.train_and_save_full_model(train_data, test_data,lstm_train_data, lstm_test_data)
        print(error_metric)
        return error_metric

    def train_and_save_full_model(self, train_data, test_data, lstm_train_data, lstm_test_data):

        # full_data.set_index('ds',inplace = True)

        prophet_model , _ = self.train_prophet(train_data=pd.concat([train_data, test_data]))
        # print(prophet_predictions, "############")

        lstm_full_data = pd.concat( [lstm_train_data,lstm_test_data] )

        lstm_train_transformed = self.std.fit_transform(lstm_full_data)
        print(lstm_train_transformed.shape)
        lstm_final_train_x, lstm_final_train_y = self.split_sequences(lstm_train_transformed, self.window_size)
        print(lstm_final_train_x.shape)

        lstm_model = self.train_lstm(lstm_final_train_x, lstm_final_train_y)

        self.save_model(prophet_model = prophet_model, lstm_model = lstm_model )

    
    def evaluate(self,actual_data, prediction_data):
        """
        This funtion evaluates the performance of the model

        Args:
            prediction_data (_type_): last row in training data
            actual_data (_type_):  true data
        """
        # measured_data = actual_data[self.data_factory.output].to_numpy()
        mae_error = mean_absolute_error(prediction_data,actual_data)
        rmse_error = np.sqrt(mean_squared_error(prediction_data,actual_data))
        print("MAE: {}, RMSE: {}".format(mae_error,rmse_error))
        # print(self.config)
        return {"RMSE": rmse_error, "MAE": mae_error}

    def save_plots(self,prediction_data,test_data, lstm_train_data):
        # print(test_data.head())
        plt.plot(lstm_train_data.index , test_data , label= "actual data")
        plt.plot(lstm_train_data.index, prediction_data, label="prediction data")
        print(self.datasource_name)
        plt.legend()
        plt.savefig(f"./save_plots/{self.datasource_name}_hybrid_lstmpredicitions.png")
        plt.show()

    def save_model(self, prophet_model, lstm_model):
        try: 
            exp_id = mlflow.create_experiment(f"{self.datasource_name} - Forecast Models")
        except:
            exp_id = mlflow.get_experiment_by_name(f"{self.datasource_name} - Forecast Models").experiment_id

        print(exp_id)
        #Run model and produce results 
        mlflow.start_run(experiment_id=exp_id, run_name=f"HybridLSTM - {self.data_factory.output} - { datetime.now().replace(second=0, microsecond=0)}")
        # TODO write model to save it in mlflow with sclar, model and params & plots.
        
        mlflow.prophet.log_model(
            prophet_model,
            artifact_path= "prophet_model",
            registered_model_name=f"{self.datasource_name}_{self.data_factory.output}_hybridlstm_prophet_model"
        )

        mlflow.pytorch.log_model(
            lstm_model,
            artifact_path= "lstm_model",
            registered_model_name=f"{self.datasource_name}_{self.data_factory.output}_hybridlstm_lstm_model"
        )

        joblib.dump(self.std, "scaler.pkl")
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
        mlflow.log_param("windows_size" , self.window_size)
        mlflow.log_param( 'dropout', self.dropout)
        mlflow.log_param("features_to_forecast", self.data_factory.output)
        mlflow.log_param("model_trained", True)
        # mlflow.log_artifact(f"./save_plots/{self.datasource_name}/lstm/predicitions.png",artifact_path="plots")
        mlflow.end_run()
   
    def create_inference_data(self, prophet_model, forecast_period):
        future_dates = prophet_model.make_future_dataframe(periods= forecast_period)
        prediction_ = prophet_model.predict(future_dates)

        y_hat = prediction_.yhat.values
        prediction_.drop('yhat',axis = 1,inplace =True)
        prediction_['y_hat'] = y_hat
        prediction_.set_index('ds',inplace = True)

        lstm_train_transformed = self.std.fit_transform(prediction_)
        print(lstm_train_transformed.shape)
        lstm_final_forecaste_x, lstm_final_train_y = self.split_sequences(lstm_train_transformed, self.window_size)
        return 

    def load_model(self):
        client = mlflow.tracking.MlflowClient("http://localhost:5000")
        prophet_model = mlflow.prophet.load_model(f"models:/{self.datasource_name}_{self.data_factory.output}_hybridlstm_prophet_model/latest")
        lstm_model = mlflow.pytorch.load_model(f"models:/{self.datasource_name}_{self.data_factory.output}_hybridlstm_lstm_model/latest")

        model_name = f'{self.datasource_name}_{self.data_factory.output}_hybridlstm_prophet_model'

        # Get the latest version of the model
        print(model_name)
        model_version = client.search_model_versions(model_name)[0].version
        print(model_name ,model_version)

        # get the metadata of the registered model
        model_details = client.get_model_version(model_name, model_version)

        print(model_details.run_id)
        client.download_artifacts(model_details.run_id, "scaler.pkl" , ".")
        scaler = joblib.load("./scaler.pkl")
        
        if os.path.exists("./scaler.pkl"):

            os.remove("./scaler.pkl")
            print("File removed: ", "scaler.pkl")

        return lstm_model , prophet_model, scaler

    def inference(self, model: MV_LSTM, test_inputs: list = None, days_ahead = None, start_date = None ,  end_date = None, from_train = False):
        """
        This function do predictions for given number of days ahead

        Args:
            model (LSTM): lstm model
            test_inputs (list): this is the last training datapoint 
            days_ahed (int): Number of days forecasts needs to be done
        """

        if start_date == None:
            start_date = datetime.now().date()
        if days_ahead == None:
            days_ahead = 50
        if end_date == None:
            end_date = start_date + timedelta(days_ahead)

        if test_inputs:
            lstm_final_test_x, lstm_final_test_y = test_inputs[0], test_inputs[1]
            print(lstm_final_test_x.shape, len(lstm_final_test_x))
        else:
            model, prophet_model, scaler = self.load_model()
            future_dates = prophet_model.make_future_dataframe(periods = days_ahead+self.window_size-1, include_history=False)
            print("predictions :", days_ahead, future_dates.shape)
            prediction = prophet_model.predict(future_dates)
            prediction.set_index('ds',inplace = True)
            scaled_test_inputs = scaler.transform(prediction)

            # scaled_test_inputs = scaled_test_inputs[:,:-1]
            lstm_final_test_x, _ = self.split_sequences(scaled_test_inputs, self.window_size)

            print(prediction, scaled_test_inputs.shape)      

        print(start_date, type(start_date), type(datetime.now().date()))
        print(days_ahead, len(lstm_final_test_x))
        model.eval()

        pred  = []
        with torch.no_grad():
            for b in range(0,days_ahead,1):
                inpt = lstm_final_test_x[b:b+1,:,:]    
                x_batch = torch.tensor(inpt,dtype=torch.float32)   
                model.init_hidden(x_batch.size(0))
                output = model(x_batch)
                print(output)
                pred.append(output.view(-1))
        
        actual_predictions = np.array(pred)

        
        days = pd.date_range(start_date, start_date + timedelta(days_ahead-1), freq='D')
        # print(lstm_final_test_x)

        preds = pd.DataFrame({'date': days, self.data_factory.output: actual_predictions})
        preds = preds.set_index('date')

        if from_train:
            print("in train ")
            dummy_data = self.lstm_test_transformed.copy()[7:,:-1]
            preds_numpy  = np.array(preds).reshape((-1,1))

            z = np.append(dummy_data,preds_numpy,axis = 1)
            rescaled_prediction_data = self.std.inverse_transform(z)[:,-1]    
        else:
            print("in test")
            preds[self.data_factory.output] = preds[self.data_factory.output].apply(lambda x: x.item())
            preds_numpy  = np.array(preds).reshape((-1,1))

            print(scaled_test_inputs.shape, preds_numpy.shape)
            z = np.append(scaled_test_inputs[:-7,:-1],preds_numpy,axis = 1)
            rescaled_prediction_data = scaler.inverse_transform(z)[:,-1]   

        preds = pd.DataFrame({'date': days, self.data_factory.output: rescaled_prediction_data})
        print(preds)
        return preds
    
    def save_data(self, data, datasource_id= None):
        if datasource_id == None:
            datasource_id = self.datasourceId



        # upload_data_to_db(data, "lstm", datasource_id)
