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
from util.helper_mlflow import getModelToServe
from worker.workflow import Workflow
from worker.forecaster_client import ForecasterClient
from data_loader.mongodb_loader import MongoLoader
from worker.forecaster_client import ForecasterClient


logging.basicConfig(
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO,
)

# logger = logging.getLogger(__name__)

alg_model_mngmt = FastAPI(
    title="AirLifeGoa Model Management Service",
    # description="Exposes apis to create. update, view and delete assets.",
    # description= "# Overview\n\niPAE is responsible for performing various activities around Industrial Assets including Condition Monitoring, Anomaly Detection, Predictive Maintenance (RUL calculation), Energy Forecasting, Asset Optimization etc.\n\nAssest configuration API service helps in managing asset configurations as per user needs. A user can able to perform following operations on assets \n```\n  1. Configure asset.\n  2. Read asset.\n  3. Update asset. \n  4. Delete assset.\n```\n\nMost of the endpoints accept `JSON` as input and return `JSON` responses.\nThis means that you must usually add the following headers to your request:\n```\nContent-type: application/json\nAccept: application/json\n```\n\n# API endpoint specification\n\n* All the requests are of type http. Additionaly each request contains a additional header called `customer-id` which represents the user-id which is used in managing the assets related that user.<br />\n* In the current version of API no authentication is being used in the API backend so no need of specifiying any authentication headers.\n\n\n# Errors\n  Asset configuration API uses following errors in responses which are defined as below \n  ### BadRequest\n  This response means that the server cannot or will not process the request due to something that is perceived to be a client error (e.g., malformed request syntax, invalid request message framing, or deceptive request routing). To resolve this, please ensure that your syntax is correct.\n  ### NotFound\n  This client error response indicates that the server cannot find the requested resource.\n  ### InternalServerError\n  This response means that issue is with the backend side where backend server or MySQL is not working properly.\n",
    version="0.0.1",
)

def custom_openapi():
    with open("./openapi.json","r") as openapi:
        return json.load(openapi)

alg_model_mngmt.openapi = custom_openapi


@alg_model_mngmt.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, _exc: RequestValidationError):
    """ Request validation error """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content="Request contains invalid data",
    )


@alg_model_mngmt.get("/", tags=["Root"], include_in_schema=False)
def hello_world():
    """Default route"""
    return "Welcome to ML Service"


@alg_model_mngmt.get("/forecast")
def forecast():
    MODEL_LIST = ["prophet"]

    for model_name in MODEL_LIST:
        model_factory = ModelFactory()
        model = model_factory.get_model_class("prophet")
        model.run_model()

@alg_model_mngmt.get("/inference/station/{id}")
def inference(id: int):
    ## TODO get bestmodel from DB currently hardcodeded
    modelFactory = ModelFactory()
    bestModel = modelFactory.get_model_class("Prophet")
    model = getModelToServe(f"{id}_Prophet_model")
    preds = bestModel.inference(model)
    return preds

   
# @alg_model_mngmt.patch("/hypertune/sensor/{id}")
# def hypertune(id: int, ):

# from mlflow.tracking import ml

# db.c



def train_model(id, model_name= "prophet_model", ):
    
    data_loader = MongoLoader()

    datapoint = data_loader.getDataPoints({"_id": id})
    
    current_model_workflow = Workflow(dataPoint=datapoint, model_name=model_name, output="PM10")
    current_model_workflow.initialize_workflow()

    print(current_model_workflow.test_data.tail(), current_model_workflow.test_data.head(), current_model_workflow.train_data.tail())

    model_client = ForecasterClient(current_model_workflow)
    model_client.train(current_model_workflow.train_data, current_model_workflow.test_data)


train_model("18")