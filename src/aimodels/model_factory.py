"""Model Type Loader Factory Class."""
import logging


from aimodels.advance_prophet import AdvanceProphet
from aimodels.customLSTM import CustomLSTM
from aimodels.customHybridLSTM import CustomHybridLSTM
from exceptions.model_exception_factory import ModelTypeError


# log = logging.getLogger(__name__)
class ModelFactory:
    """Model Factory Class."""
    def __init__(self):
       pass
    @staticmethod
    def get_model_class(model_type, *kwargs):
        """
        Return Model Class object based on the selected model type. This is a static method.

        :param model_type: Name of the type of  model
        :param ml_params: Parameters for initializing the model.
        :param target: Target for model
        :return: Model Class instance
        """
        model_class = None
        if model_type.lower() == "prophet":
            model_class = AdvanceProphet(*kwargs)
        elif model_type.lower() == "lstm":
            model_class = CustomLSTM(*kwargs)
        elif model_type.lower() == "hybridlstm":
            model_class = CustomHybridLSTM(*kwargs)
        else:
            raise ModelTypeError(
                "Model type "
                + model_type
                + " not defined. Please check the model name and retry"
            )

        return model_class
