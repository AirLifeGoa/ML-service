import schedule
import time

from pymongo import MongoClient

client = MongoClient('mongodb://localhost',27017)
db = client["pollution"]
data_logs = db["datalogs"]

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
    granularity = dataSource["granularity"]

    if granularity == "hours" and newDataPoints>=24:
        return True
    if granularity == "daily" and newDataPoints>=5:
        return True                
    if granularity == "seconds" and newDataPoints>= 120:
        return True

    return False


# Define a function to retrieve data from the database
def fetch_datalogs_from_db():
    dataSources = list(data_logs.find({}))
    for dataSource in dataSources:
        if check_train(dataSource):
            #make apinrequest
            pass

    # print(result)


# Schedule the job to run every 15 minutes
# schedule.every(1).minutes.do(fetch_datalogs_from_db)

schedule.every(10).seconds.do(fetch_datalogs_from_db)

# Run the scheduled job
while True:
    schedule.run_pending()
    time.sleep(1)
