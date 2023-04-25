from pymongo import MongoClient
import sys
sys.path.append(r'C:\Users\hp\Desktop\ML server\src')
import numpy as np
import pandas as pd
import sys, os
import re
from pathlib import Path
from  datetime import datetime
from db_config.settings import mongodb_uri, port, pollutionDB, dataPointsCollection, pollutionDataCollection, modelLogs


class MongoLoader:
    
    def __init__(self, client = None, db = pollutionDB, dataPointsCollection = dataPointsCollection, pollutionfdata = pollutionDataCollection, modellogs= modelLogs):
        self.client =client
        self.db_name = db
        self.dataPoints = dataPointsCollection
        self.pollutiondata = pollutionDataCollection
        self.modelLogs= modelLogs
        self.connect_client()
    
    def connect_client(self):
        
        self.client = MongoClient(mongodb_uri, port)
        print(self.dataPoints)
        self.db_connection = self.client[self.db_name]
        self.datasource_connection = self.db_connection[self.dataPoints]
        self.data_connection = self.db_connection[self.pollutiondata]
        self.modelLogs_connection = self.db_connection[self.modelLogs]

    def getDataPoints(self, query={}):     
        print(self.dataPoints)
        documents = list(self.datasource_connection.find(query))
        print(documents)
        return documents[0]

    def getDataAllSources(self):

        try:
            dataSources = list(self.datasource_connection.find({}))

            dataSourcesList = []

            for source in dataSources:
                dataSourcesList.append(source["name"])
            
            return dataSourcesList
        except:
            return []
    
    def update_model_logs(self, id, lastTrainingPoint, metric_errors, output):
        bestModel = min(zip(metric_errors.values(), metric_errors.keys()))[1]
        metadata = {"dataSourceId": id, "lastTrainingPoint":lastTrainingPoint, "metric": output}
        data = { "recordedAt": datetime.today().replace(microsecond=0), "metadata": metadata, "bestModel": bestModel, "data": metric_errors}
        self.modelLogs_connection.insert_one(data)

    def update_existing_logs(self,id, lastTrainingPoint, existing_log,metric_errors, output):

        full_error_metric = {**metric_errors, **existing_log["data"]}
        bestModel = min(zip(full_error_metric.values(), full_error_metric.keys()))[1]
        print(full_error_metric)
        query = {"metadata.dataSourceId": id, "metadata.lastTrainingPoint": lastTrainingPoint, "metadata.metric": output}
        # query = [
        #     {'$match': {
        #             'metadata.datasourceId': id, 
        #             'metadata.metric': output,
        #             'metadata.lastTrainingPoint': lastTrainingPoint
        #         }
        #     },
        #       {
        #     '$sort': {
        #         'recordedAt': -1
        #         }
        #     }, {
        #         "$limit" : 1
        #     }
        # ]
        upatedvalues = [{ "$set": { "data": full_error_metric, "bestModel": bestModel } }]
        result = self.modelLogs_connection.update_many(query, upatedvalues)
        print(result.raw_result)
                            
    def load_data(self,id):

        metrics = self.getDataPoints({"_id":str(id)})["metrics"]
        unpack_data ={"date":"$recordedAt"}
        for metric in metrics:
            metric = re.sub(r"[^a-zA-Z0-9]","",metric)
            unpack_data[metric] = f"$data.{metric}"
        
        print(unpack_data)
        pipeline = [
            {"$match": {"metadata.dataSourceId": str(id)}},
            {"$sort": { 'metadata.addedAt': -1, }},
            {"$group": { "_id": { "recordedAt": '$recordedAt', "sourceId": '$metadata.sourceId', }, "data": { "$first": '$$ROOT',},},},
            {"$replaceRoot": { "newRoot": '$data', },},
            {"$sort": { "recordedAt": 1,},},
            {"$project": {"dataSourceId": False, "metadata": False, "__v": False, "_id": False, "uploadedBy": False}},
            {"$project": unpack_data, },
            # {
            #     "$project": {
            #         "date": {
            #             "$convert": {
            #             "input": "$recordedAt",
            #             "to": "date",
            #             }
            #         }
            #     }
            # }
            # {"$cond": [ { "$eq": ["$field", "value"] },"$filed",np.nan] }
            {
        '$addFields': {
            'date': {
                '$add': ['$date', 330*60*1000]
            }
        }
    }
        ]
        result = list(self.data_connection.aggregate(pipeline))

        result = result[:-1]
        
        return self.convert_data_to_df(result)

    def convert_data_to_df(self,data):
        print(data)
        df = pd.DataFrame.from_dict(data)
        df = df.set_index("date")
        return df

if __name__ == "__main__":
    obj = MongoLoader()
    df = obj.load_data(18)
    print(df.tail())
