import logging
import warnings

from aimodels.model_factory import ModelFactory
from worker.workflow import Workflow
from data_loader import mongodb_loader


class ForecasterClient:

    def __init__(self, datapoint, model_type, output):
        """
        Initialize forecaster.

        :param target: Name of the target for which forecaster is being initialized.
        :param config: Configuration for the forecaster.
        """
        self.workflow = None
        self.dataPoint = datapoint
        self.model_type = model_type
        self.output_config = output
        

        print("Forecaster Client initiated")

    def initialize_trainer(self):
        """
        Initialize the Trainer object.

        :returns: the trainer object after creation
        """
        self.model_client = ModelFactory.get_model_class(
            self.model_type, self.workflow.config,  self.dataPoint["_id"], self.dataPoint["name"], self.workflow
        )

        self.trainer = self.model_client
        return self.trainer

    def train(self):
        """
        Train the model.

        :returns: model validation metrics.
        """
        metrics = None
        metrics = self.trainer.train(self.workflow.train_data, self.workflow.test_data)

        # if model is not None and scaler is not None:
        #     print("Model loaded successfully")
        #     metrics = self.trainer.validate()
        return metrics

    def initialize_workflow(self):
        """
        Initialize the Predictor object.

        :returns: Boolean value model_loaded, whether model is loaded or not.
        """
        self.workflow = Workflow(
            self.dataPoint, self.model_type, self.output_config
        )

        self.workflow.initialize_workflow()
        self.initialize_trainer()

        ##based on load_config check whether model already exists in mlflow or not.
        ## if exists return true - do not train model only retrain
        ##else return false - only train the model.


