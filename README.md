
# AirLifeGoa - ML Server  

This project aims to forecast pollution levels (PM10, PM2.5 & gases) using AI models, including Long Short-Term Memory (LSTM), Facebook's Prophet, and a Hybrid model of both. The project uses Python FastAPI to provide a web service that exposes the pollution forecasts.

# Introduction
Air pollution is a critical environmental problem that poses a significant risk to human health. Accurate forecasting of pollution levels is crucial to help take preventive measures to reduce the impact of pollution on the environment and human health. This project aims to forecast pollution levels in Goa using AI models, including LSTM, Prophet, and a hybrid model of both, and expose the forecasts through a FastAPI web service. Currently, this server forecasts pollution levels of 19 stations in Goa.





## API Architecture
The figure above shows the architecture of the ML Server. The ML Server is written in Python and it uses Pollution DB and MLflow service for easy serving of models. The FastApi framework is used for developing the ML Server.
![ML Server Architecture + Workflow ](https://github.com/AirLifeGoa/ML-service/blob/main/architecture_workflow.png)
The image describes the overall architecture of the server. There is a cron job running in the server called DynamicModelTrainer which queries the pollution DB at regular intervals. If there is any new data available to train the model, the DynamicModelTrainer immediately makes an API request to the server to train the model for a particular data source.

At present, the server has only two API routes: Training and Inference. The API documentation is given [here](#API-Reference). 

### Forecast workflow
  The forecast workflow basically starts by initializing a worker which initializes the forecast client for a particular data source and metric it wants to forecast. The forecast client's first step is to load data from the DB. Then data preprocessing happens, such as dropping NaN values, scaling the data, etc. The preprocessing steps for each model are written in src/data_factory. Therefore, implementing new methods or creating new data prep processors for new models can be done in src/data_factory. Once the forecast client is ready, model training starts and it has two phases:

First, the model is trained on the training dataset and tested on the testing dataset. Using testing accuracy, the best model is saved in the DB whose forecasts are further used for showing data in the frontend dashboard.
The entire dataset is then given to train the model, and this model is stored in MLflow for making forecasts. This is because in time series, the forecasts will be more accurate with more recent data.
As the final steps, the entire model logs, including errors, parameters, and models, are saved in MLflow, and the predictions are saved in the database.

### Inference workflow
 Making forecasts is straightforward. The first step is to initialize the inference client, which takes care of any test input data requirements. Then it loads the scaler (if used during model training) and the model from MLflow. Afterward, forecasts are made for a given time period, and predictions are saved in the DB.
## Installation

Conda Environment setup 

```
  conda create -n <envname> python=3.8 anaconda
```
Install ML-server with python 3.8

```bash
  pip install -r requirements.txt
  cd src
  uvicorn main:alg_model_mngmt --port 8000
```

To spin up mlflow server 
```
mlflow server --default-artifact-root file:///./mlruns --serve-artifacts -h 0.0.0.0  
```

Navigate to ```<domain>:8000/docs``` for accesing fastapi swagger docui for training & doing inference in models.

Navigate to ```<domain>:5000/docs``` for accesing all the data related to trained models like hyperparameters, errors, plots, scaler etc.



## API Reference

#### Training the models

```http
  POST /forecast
```
  This API calls is used for training the models manually.

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `model_name` | `string` | **Required** Select the model to train |
| `metric` | `string` | **Required** Select the metric to train model |
| `station` | `string` | **Required** Select the datasource to train model |

#### Pollution Forecasting


```http
  POST /inference
```
This API calls is used for making forecasts on the trained models.

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `start_date`      | `string` | **Required** Forecasting start date in YYYY-MM-DD format|
| `end_date`      | `string` | **Required** Forecasting end date in YYYY-MM-DD format|
| `model_name` | `string` | **Required** Select the model to train |
| `metric` | `string` | **Required** Select the metric to train model |
| `station` | `string` | **Required** Select the datasource to train model |
| `save_predictions`      | `string` | **Required** Boolean value, saves predictions to DB if set true |

## Contributing

We welcome contributions of any size! please follow the below link of google docs to get more info about taking on new tasks.

Future Extension(#)

Got stuck. For any more info contact us

Divyansh K - 6204991800 or mail at - me@divyanx.com        
Pranav B - 9390062480  or mail at - pranavbalijapelly@gmail.com

