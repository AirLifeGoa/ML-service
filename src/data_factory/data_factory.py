"""Model Type Loader Factory Class."""
import logging


from data_factory.prophet_data import ProphetData
from data_factory.lstm_data import LSTMData
from data_factory.hybridlstm_data import HybridLSTMData
from exceptions.model_exception_factory import ModelTypeError

# log = logging.getLogger(__name__)


class ModelDataFactory:
    """Model Factory Class."""

    @staticmethod
    def get_model_data_class(model_type, **kwargs):
        """
        Return Model Class object based on the selected model type. This is a static method.

        :param model_type: Name of the type of  model
        :param ml_params: Parameters for initializing the model.
        :param target: Target for model
        :return: Model Class instance
        """
        model_class = None
        if model_type.lower() == "prophet":
            model_class = ProphetData(**kwargs)
        elif model_type.lower() == "lstm":
            model_class = LSTMData(**kwargs)
        elif model_type.lower() == "hybridlstm":
            model_class = HybridLSTMData(**kwargs)
        else:
            raise ModelTypeError(
                "Model type "
                + model_type
                + " not defined. Please check the model name and retry"
            )

        return model_class
