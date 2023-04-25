import logging
import json
import sys
from aimodels.model_factory import ModelFactory
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from db_config import database
from db_config.database import db
from fastapi import FastAPI
import mlflow
from mlflow.tracking import MlflowClient
from bson import ObjectId
from aimodels.model_factory import ModelFactory
from datetime import datetime
# from util.helper_mlflow import getModelToServe
from worker.workflow import Workflow
from worker.forecaster_client import ForecasterClient
from data_loader.mongodb_loader import MongoLoader
from worker.forecaster_client import ForecasterClient
from worker.inference_client import InferenceClient


def train_model(id, model_name= "hybridlstm", ):
    
    data_loader = MongoLoader()

    datapoint = data_loader.getDataPoints({"_id": id})
    
    print(datapoint)

    ##get output from config file
    # current_model_workflow = Workflow(dataPoint=datapoint, model_name=model_name, output="PM10")
    # current_model_workflow.initialize_workflow()

    # print(current_model_workflow.train_data[-1])
    # print(current_model_workflow.config)
    model_client = ForecasterClient(datapoint, model_name,output="PM10")
    model_client.initialize_workflow()
    # model_client.workflow
    model_client.workflow.update_model_logs({},"PM10")
    # error_metric = model_client.train()
    # error_metric =10
    # print(error_metric)

    # start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    # end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    
    # inference_client = InferenceClient(datapoint, model_name, output="PM10")
    # inference_client.initialize_workflow()
    # predictions = inference_client.forecast(start_date,)
    # inference_client.model_client.save_data(predictions, id)

    return ""

train_model("6")