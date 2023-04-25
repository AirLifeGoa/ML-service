from os import path
from datetime import datetime
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
        self.last_training_point = self.data.index[-1]

        self.data_factory = ModelDataFactory()
        self.data_processor = self.data_factory.get_model_data_class(self.model_name)
        self.train_data, self.test_data = self.data_processor.prepare_data(self.data, self.output)

    def update_model_logs(self, metric, output):
        self.data_loader = MongoLoader()
        model_logs_conn = self.data_loader.modelLogs_connection

        print({"metadata.dataSourceId": self.datapointId, "metadata.lastTrainingPoint": self.last_training_point, "metadata.metric":output})
        # result = list(model_logs_conn.find({"metadata.dataSourceId": self.datapointId, "metadata.lastTrainingPoint": self.last_training_point, "metadata.metric":output}))
        result = list(model_logs_conn.aggregate([
            {'$match': {
                    'metadata.dataSourceId': self.datapointId, 
                    'metadata.metric': output,
                    'metadata.lastTrainingPoint': { "$eq":datetime.utcfromtimestamp(self.last_training_point.timestamp())}
                }
            },
              {
            '$sort': {
                'recordedAt': -1
                }
            }, {
                "$limit" : 1
            }
        ]))
        print("datalogs result ", result)
        if len(result) ==0:
            self.data_loader.update_model_logs(id=self.datapointId, lastTrainingPoint=self.last_training_point, metric_errors=metric, output=output)
        else:
            self.data_loader.update_existing_logs(self.datapointId,self.last_training_point, result[0], metric , output)


if __name__ == "__main__":

    datapoint = MongoLoader()
    data= datapoint.getDataPoints({"_id":"6"})

    obj = Workflow(data,"prophet_model","PM10")
    obj.update_model_logs({},"PM10")