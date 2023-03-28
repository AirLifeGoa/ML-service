import logging
import warnings

from aimodels.model_factory import ModelFactory
from worker.workflow import Workflow
from util.helper_mlflow import MLflowUtils

class InferenceClient:

    def __init__(self,  datapoint, model_type, output):
        self.workflow = None
        self.dataPoint = datapoint
        self.model_type = model_type
        self.output_config = output
        

        print("Inference Client initiated")

    def initialize_client(self):

        self.model_client = ModelFactory.get_model_class(
            self.model_type, self.workflow.config,  self.dataPoint["_id"], self.dataPoint["name"], self.workflow
        )

        self.trainer = self.model_client
        return self.trainer
    
    def forecast(self, start_date, end_date, forecast_period):

        model = MLflowUtils().load_model(self.dataPoint['name'] ,self.model_type, self.output_config)
        
        return self.model_client.inference(model, self.workflow.data_processor.test_inputs, days_ahead = forecast_period,  start_date= start_date, end_date = end_date)



    def initialize_workflow(self):
        """
        Initialize the Predictor object.

        :returns: Boolean value model_loaded, whether model is loaded or not.
        """
        self.workflow = Workflow(
            self.dataPoint, self.model_type, self.output_config
        )

        self.workflow.initialize_workflow()
        self.initialize_client()

        
