"""Parser for various configuration files."""
import logging
import os
from datetime import datetime, timedelta
from typing import List

import yaml

# log = logging.getLogger(__name__)


class AppConfigParser:
    """Application Config Parser."""

    def __init__(
        self, config_file_path
    ):
        """
        Initialize application config parser.

        :param file_name: Configuration file name.
        """
        if os.path.isfile(config_file_path):
            # log.status("Config file found")
            with open(config_file_path) as file:
                self.config = yaml.load(file, Loader=yaml.FullLoader)
            
            print(self.config)
                
        # else:
        #     log.error("Config file not found")

    def parse_ml_parameters(self):
        """
        Parse ML parameters required for modeling.

        :returns: None
        """
        model = self.config["model"]
        self.model_type = model["model_type"]
        self.target = model["target"]
        ml_params = model["model_params"]
        self.ml_params = ml_params
        self.epochs = ml_params["epochs"]
        self.nsteps = ml_params["nsteps"]
        self.ml_params.update(
            {
                "features": self.input_features,
                "mlp_features": self.mlp_features,
                "timeseries_features": self.timeseries_features,
                "features_to_forecast": self.features_to_forecast,
                "api": False,
            }
        )

    def change_params(
        self,
        model_type, **kwargs
    ):
        """Modify params for AppConfigParser."""
        if model_type:
            self.model_type = model_type
            if self.model_type == "prophet":
                self.ml_params = self.prepare_prophet_params(**kwargs)
        

    def prepare_prophet_params(self, **kwargs):

        # if days_ahead:
        #     self.forecast_period = days_ahead
        # if input_features:
        #     self.ml_params["features"] = input_features
        #     self.ml_params["timeseries_features"] = input_features
        #     if model_type == 'ann':
        #         self.ml_params['mlp_features'] = input_features
        # # if timeseries_features:
        # #     self.ml_params["timeseries_features"] = timeseries_features
        # # if mlp_features:
        # #     self.ml_params["mlp_features"] = mlp_features
        # if features_to_forecast:
        #     self.ml_params["features_to_forecast"] = features_to_forecast
        # if file_path:
        #     self.file_path = file_path
        # if file_name:
        #     self.file_name = file_name
        # if target:
        #     self.target = target
        # if nsteps:
        #     self.ml_params["nsteps"] = nsteps
        # if model_folder:
        #     self.ml_params["model_file_path"] = model_folder
        # if data_source:
        #     self.data_source = data_source
        pass
    def get_params(self):
        """Get all the parameters for AppConfigParser."""
        return vars(self)

