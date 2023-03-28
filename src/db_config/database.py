from pymongo import MongoClient
from db_config import settings

client = MongoClient(settings.mongodb_uri, settings.port)
db = client[settings.pollutionDB]
print(db.list_collection_names())
if "predictiondatas" not in db.list_collection_names():
    connection = db.create_collection('predictiondatas', timeseries={ 'timeField': 'recordedAt', 'metaField': 'metadata', 'granularity': 'hours' })
    
else:
    connection = db.get_collection('predictiondatas')
# db.client