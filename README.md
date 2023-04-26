
# AirLifeGoa - ML Server  

This project aims to forecast pollution levels(PM10, PM2.5 & gases) using AI models, including Long Short-Term Memory (LSTM), FAcebook's Prophet, and a Hybrid model of both. The project uses Python FastAPI to provide a web service that exposes the pollution forecasts.

# Introduction
Air pollution is a critical environmental problem that poses a significant risk to human health. Accurate forecasting of pollution levels is crucial to help take preventive measures to reduce the impact of pollution on the environment and human health. This project aims to forecast pollution levels in Goa using AI models, including LSTM, Prophet, and a hybrid model of both, and expose the forecasts through a FastAPI web service. Currently this server forcasts pollution levels of 19 stations in Goa.





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

