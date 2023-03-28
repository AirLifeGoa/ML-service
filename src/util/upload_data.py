from db_config.database import connection
from  datetime import datetime


#while uploading prediction data to database think it like this. we will have datasourcd id -> train/retrain a model it for every possible output configd metioned in the config file. 
# then collect cummulative prediction data for all inputs. then send all data at once to upload data function.  

#add extra field in metadata or normal field called best metric  which gives best mertic among all models.
def upload_data_to_db(dataframe, model_name, id):
    
    columns = dataframe.columns
    data_to_upload = []
    for index,prediction_row in dataframe.iterrows():
        pred_datapoint = { "recordedAt":index,"addedAt": datetime.today().replace(microsecond=0), "data":{}, "metadata": {"modelName": model_name, "dataSourceId": id} }
        for col in columns:
            pred_datapoint["data"][col]= prediction_row[col]
        print(pred_datapoint)
        data_to_upload.append(pred_datapoint)
   
    connection.insert_many(data_to_upload)
    