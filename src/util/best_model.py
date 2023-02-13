from pymongo import MongoClient
from db_config import settings


class BestModel:

    def __init__(self):
        client = MongoClient(settings.mongodb_uri, settings.port)
        db = client['MlModels']
        print(db.list_collection_names())
        if "modelPredictions" not in db.list_collection_names():
            self.connection = db.create_collection('bestmodels', timeseries={ 'timeField': 'timestamp', 'metaField': 'data', 'granularity': 'hours' })    
        else:
            self.connection = db.get_collection('bestmodels')
    
    def upload_data(self, timestamp, best_model):
        # try: 
            self.connection.insert_one({"timestamp" : timestamp,   "metadata" : { "bestmodel": best_model} } )
        # except:
            # print("unable to upload data")
            # logging.info("unable to upload data")
            return