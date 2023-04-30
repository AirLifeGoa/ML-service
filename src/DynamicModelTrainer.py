import schedule
import time
import requests
from pymongo import MongoClient
from main import ForecastInput, train_model

client = MongoClient('mongodb+srv://divyanx:airlifegoa@cluster0.hbuq2jq.mongodb.net/?retryWrites=true&w=majority',27017)
db = client["test"]
data_logs = db["datalogs"]
datapoints =  db["datapoints"]

""" datalogs Schema
    - new data points
    - previous trained time
    - last updated (Data)
    - granularity of data
    - 
"""


def check_train(dataSource):
    id = dataSource["_id"]
    lastUpdated = dataSource["lastUpdated"]
    previousTrainedTime = dataSource["previousTrainedTime"]
    newDataPoints = dataSource["newDataPoints"]
    
    dataSource = list(datapoints.find({"_id": id}))
    granularity = dataSource[0]["expectedFrequencyType"] 

    if granularity == "hours" and newDataPoints>=24:
        return True
    if granularity == "days" and newDataPoints>=5:
        return True                
    if granularity == "seconds" and newDataPoints>= 120:
        return True

    return False

MODELS = ["prophet", "lstm","hybridlstm"]
METRICS = ["PM10", "PM25"]

# Define a function to retrieve data from the database
def fetch_datalogs_from_db():
    dataSources = list(data_logs.find({}))
    for dataSource in dataSources:
        if check_train(dataSource):
            datapoint = datapoints.find({"_id":dataSource["_id"]})
            station_name = datapoint[0]["name"]
            for model in MODELS:
                for metric in METRICS:
                    payload = { "model_name": model, "station":station_name, "metric": metric}
                    payload = ForecastInput()
                    payload.metric = metric
                    payload.model_name = model
                    payload.station  = station_name
                    response  = train_model( data=payload)

                    # response.raise_for_status()  # This will raise an error if the request fails
                    print(response)
        
        query = {"_id": dataSource["_id"]}
        upatedvalues = [{ "$set": { "newDataPoints": 0 } }]
        data_logs.update_one(query,upatedvalues)
        break

    # print(result)


# Schedule the job to run every 15 minutes
# schedule.every(1).minutes.do(fetch_datalogs_from_db)

schedule.every(10).seconds.do(fetch_datalogs_from_db)

# Run the scheduled job
while True:
    schedule.run_pending()
    time.sleep(1)
