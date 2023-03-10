from pymongo import MongoClient
import numpy as np
import pandas as pd
import sys, os
from pathlib import Path

from db_config.settings import mongodb_uri, port, pollutionDB, dataPointsCollection, pollutionDataCollection


class MongoLoader:
    
    def __init__(self, client = None, db = pollutionDB, dataPointsCollection = dataPointsCollection, pollutionfdata = pollutionDataCollection):
        self.client =client
        self.db_name = db
        self.dataPoints = dataPointsCollection
        self.pollutiondata = pollutionDataCollection
        self.connect_client()
    
    def connect_client(self):
        
        self.client = MongoClient(mongodb_uri, port)
        print(self.dataPoints)
        self.db_connection = self.client[self.db_name]
        self.sensor_connection = self.db_connection[self.dataPoints]
        self.data_connection = self.db_connection[self.pollutiondata]

    def getDataPoints(self, query={}):     
        print(self.dataPoints)
        documents = list(self.sensor_connection.find(query))
        print(documents[0])
        return documents[0]
                            
    def load_data(self,id):

        metrics = self.getDataPoints({"_id":str(id)})["metrics"]
        unpack_data ={"date":"$recordedAt"}
        for metric in metrics:
            unpack_data[metric] = f"$data.{metric}"

        pipeline = [
            {"$match": {"metadata.dataSourceId": str(id)}},
            {"$sort": { 'metadata.addedAt': -1, }},
            {"$group": { "_id": { "recordedAt": '$recordedAt', "sourceId": '$metadata.sourceId', }, "data": { "$first": '$$ROOT',},},},
            {"$replaceRoot": { "newRoot": '$data', },},
            {"$sort": { "recordedAt": 1,},},
            {"$project": {"dataSourceId": False, "metadata": False, "__v": False, "_id": False, "uploadedBy": False}},
            {"$project": unpack_data, }
            # {
            # "$project": {
            #     "date": {
            #         "$convert": {
            #         "input": "$date",
            #         "to": "timezone",
            #         "timezone": "Asia/Kolkata"
            # }
#          }
#       }
#    }
            # {"$cond": [ { "$eq": ["$field", "value"] },"$filed",np.nan] }
        ]
        result = list(self.data_connection.aggregate(pipeline))
        
        return self.convert_data_to_df(result)

    def convert_data_to_df(self,data):

        df = pd.DataFrame.from_dict(data)
        df = df.set_index("date")
        return df

if __name__ == "__main__":
    obj = MongoLoader()
    obj.load_data(18)
