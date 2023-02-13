# import time
# import platform
# from argparse import ArgumentParser
# import mlflow
# from mlflow.entities import Param,Metric,RunTag

# mlflow.set_tracking_uri("http://localhost:5000")
# print("MLflow Version:", mlflow.__version__)
# print("Tracking URI:", mlflow.tracking.get_tracking_uri())
# client = mlflow.tracking.MlflowClient("http://localhost:5000")

# experiment=client.create_experiment("anomaly_detection")

# run=client.create_run(experiment)
# client.log_dict(run.info.run_id,{"input_features":["Leaving Chilled Water Temperature","Return Chilled Water Temperature"]},"input.json")

# import mlflow
import os, mlflow
from mlflow.tracking import MlflowClient
client = MlflowClient("http://localhost:5000")
# client.get
print(client.get_experiment_by_name("Default"))
print(mlflow.artifacts.download_artifacts)
# client.get 

# local_dir = "./new"
# if not os.path.exists(local_dir):
#   os.mkdir(local_dir)

# # Creating sample artifact "features.txt".
# features = "rooms, zipcode, median_price, school_rating, transport"
# with open("features.txt", 'w') as f:
#     f.write(features)

# Creating sample MLflow run & logging artifact "features.txt" to the MLflow run.
# with mlflow.start_run(experiment_id="0") as run:
#     mlflow.log_artifact("features.txt", artifact_path="features")
#     mlflow.set_tag("1","2")

# # Download the artifact to local storage.
# local_path = client.download_artifacts("dd68c3dde7344e239e9f3fea824eea97", "features", local_dir)
# print("Artifacts downloaded in: {}".format(local_dir))
# print("Artifacts: {}".format(local_dir))
client.log_d

# # def run(alpha, run_origin):
# #     with mlflow.start_run(run_name=run_origin) as run:
# #         print("runId:",run.info.run_id)
# #         print("experiment_id:", run.info.experiment_id)
# #         print("experiment_name:", "new_exp")
# #         print("artifact_uri:", mlflow.get_artifact_uri())
# #         print("alpha:", alpha)
# #         print("run_origin:", run_origin)
# #         mlflow.log_param("alpha", alpha)
# #         mlflow.log_metric("rmse", 0.789)
# #         mlflow.set_tag("name", "exp")
# #         # mlflow.set_tag("run_origin", run_origin)
# #         mlflow.set_tag("version.mlflow", mlflow.__version__)
# #         mlflow.set_tag("version.python", platform.python_version())
# #         mlflow.set_tag("version.platform", platform.system())
# #         mlflow.log_dict({"input_features":["Leaving Chilled Water Temperature","Return Chilled Water Temperature"]},"input.json")
# #         mlflow.log_dict({"output_features":["Leaving Chilled Water Temperature","Return Chilled Water Temperature"]},"output.json")
# #         with open("info.txt", "w") as f:
# #             f.write("Hi artifact")
# #         mlflow.log_artifact("info.txt")
# #         params = [ Param("p1","0.1"), Param("p2","0.2") ]
# #         now = round(time.time())
# #         metrics = [ Metric("m1",0.1,now,0), Metric("m2",0.2,now,0) ]
# #         tags = [ RunTag("tag1","hi1"), RunTag("tag2","hi2") ]
# #         client.log_batch(run.info.run_id, metrics, params, tags)

# # if __name__ == "__main__":
# #     parser = ArgumentParser()
# #     parser.add_argument("--experiment_name", dest="experiment_name", help="Experiment name", default=None, type=str)
# #     parser.add_argument("--alpha", dest="alpha", help="alpha", default=0.1, type=float )
# #     parser.add_argument("--run_origin", dest="run_origin", help="run_origin", default="")
# #     args = parser.parse_args()
# #     print("Arguments:")
# #     for arg in vars(args):
# #         print(f"  {arg}: {getattr(args, arg)}")
# #     if args.experiment_name is not None:
# #         mlflow.set_experiment(args.experiment_name)
# #     run(args.alpha,args.run_origin)

# # # from minio import Minio
# # # import os
# # # # LOCAL_FILE_PATH = os.environ.get('LOCAL_FILE_PATH')
# # # # ACCESS_KEY = os.environ.get('ACCESS_KEY')
# # # # SECRET_KEY = os.environ.get('SECRET_KEY')
# # # MINIO_API_HOST = "http://localhost:9000"
# # # MINIO_CLIENT = Minio("localhost:9000", access_key='admin', secret_key='sample_key', secure=False)
# # # def main():
# # #     found = MINIO_CLIENT.bucket_exists("mlflow")
# # #     if not found:
# # #        MINIO_CLIENT.make_bucket("mlflow")
# # #     else:
# # #        print("Bucket already exists")
# # #     MINIO_CLIENT.fput_object("mlflow", "<pyfile.py>", "C:\\Users\\hp\\Desktop\\new_mlflow_api\\mlflow-docker-master\\create_bucket.py")
# # #     print("It is successfully uploaded to bucket")

# # # # main()


# from fastapi import FastAPI


# app= FastAPI()

# @app.get("/")
# def get():
#     return "hello world"
