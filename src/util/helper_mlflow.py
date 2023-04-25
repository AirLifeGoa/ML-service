import mlflow
import sys
sys.path.append(r'C:\Users\hp\Desktop\ML server\src')

MLFLOW_HOST = "localhost:"
MLFLOW_PORT = "5000"


class MLflowUtils():

    def __init__(self, mlflow_host=None, mlflow_port=None):
        self.mlflow_uri = "http://"
        if (mlflow_host):
            self.mlflow_uri += mlflow_host 
        else:
            self.mlflow_uri += MLFLOW_HOST
        
        if mlflow_port:
            self.mlflow_uri += mlflow_port
        else:
            self.mlflow_uri += MLFLOW_PORT

    def get_model_flavour(self, model_type):
        if model_type == "prophet":
            return mlflow.prophet
        elif model_type == "lstm":
            return mlflow.pytorch
        else:
            raise Exception(" No flavour named ",model_type)
    def load_model(self, datasource_name, model_type, output):
        print(model_type)
        try:
            self.model_uri = f"models:/{datasource_name}_{output}_{model_type}/latest"
            mlflow.set_tracking_uri(self.mlflow_uri)
            model_flavor = self.get_model_flavour(model_type)
            model = model_flavor.load_model(model_uri=f"models:/{datasource_name}_{output}_{model_type}/latest")
            print(model)
            return model
        except Exception as err:
            print(err)
            return "Unable to load model"
        
    # def load_scalar()
    


if __name__ == "__main__":
    obj = MLflowUtils()
    obj.get_model_to_serve("iitgoa-hostel-3", "lstm", "PM10")