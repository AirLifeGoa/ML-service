from os import path
from util.modelConfig import ModelConfigClient
from data_factory.data_factory import ModelDataFactory
from data_loader.mongodb_loader import MongoLoader

class Workflow:

    def __init__(self, dataPoint, model_name, output):
        self.datapointId = dataPoint["_id"]
        self.datapoint_name = dataPoint["name"]
        self.model_name = model_name
        self.output  = output
        self.datapoint = dataPoint


    def initialize_workflow(self):
        self.load_config()
        self.load_data()

    def load_config(self):
        self.config_client = ModelConfigClient()
        self.config, loaded = self.config_client.load_model_config(self.datapointId,self.model_name)
        print("self.config ", self.config)

        return loaded
        # self.config = self.config[self.model_name]
    
    def load_data(self):

        self.data_loader = MongoLoader()
        self.data = self.data_loader.load_data(self.datapointId)

        self.data_factory = ModelDataFactory()
        self.data_processor = self.data_factory.get_model_data_class(self.model_name)
        self.train_data, self.test_data = self.data_processor.prepare_data(self.data, self.output)


if __name__ == "__main__":

    datapoint = MongoLoader()
    data= datapoint.getDataPoints({"_id":"18"})

    obj = Workflow(data,"prophet_model","PM10")
    obj.load_data()