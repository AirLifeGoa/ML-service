import mlflow

def getModelToServe(modelName,metric=None,min=True,tracking_uri="http://localhost:5000"):
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        model_flavor = mlflow.pyfunc.load_model(model_uri=f"models:/{modelName}/{1}").metadata.flavors
        if model_flavor[list(model_flavor.keys())[0]]["model_type"] == "Prophet":
            model = mlflow.prophet.load_model(model_uri=f"models:/{modelName}/{1}")
        
        return model
    except Exception as err:
        print(err)
        return "Unable to load model"
        