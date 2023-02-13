import logging
import warnings

from aimodels.model_factory import ModelFactory
from worker.workflow import Workflow


class ForecasterClient:

    def __init__(self, workflow: Workflow):
        """
        Initialize forecaster.

        :param target: Name of the target for which forecaster is being initialized.
        :param config: Configuration for the forecaster.
        """
        self.workflow = workflow
        self.dataPoint = workflow.datapoint
        self.model_type = workflow.model_name
        self.output_config = workflow.output
        self.initialize_trainer()

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

    def train(self, train_data, test_data):
        """
        Train the model.

        :returns: model validation metrics.
        """
        metrics = None
        model = self.trainer.train(train_data, test_data)

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


