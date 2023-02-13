from db_config.database import connection


def upload_data_to_db(dataframe, model_name, id):
    
    columns = dataframe.columns
    data_to_upload = []
    for index,prediction_row in dataframe.iterrows():
        pred_datapoint = {"timestamp":index, "metadata": {"model_name": model_name, "type": "station", "id": id} }
        for col in columns:
            pred_datapoint[col]= prediction_row[col]
        print(pred_datapoint)
        data_to_upload.append(pred_datapoint)
   
    connection.insert_many(data_to_upload)
    