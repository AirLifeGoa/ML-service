from pymongo import MongoClient
from db_config import settings

client = MongoClient(settings.mongodb_uri, settings.port)
db = client['MlModels']
print(db.list_collection_names())
if "modelPredictions" not in db.list_collection_names():
    connection = db.create_collection('modelPredictions', timeseries={ 'timeField': 'timestamp', 'metaField': 'data', 'granularity': 'hours' })
    
else:
    connection = db.get_collection('modelPredictions')
# db.client