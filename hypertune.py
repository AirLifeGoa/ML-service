import hyperopt

def build_train_objective(x_train: Union[pd.DataFrame, np.array],
                          y_train: Union[pd.Series, np.array],
                          x_test: Union[pd.DataFrame, np.array],
                          y_test: Union[pd.Series, np.array],
                          metric: str):
    """Build optimization objective function fits and evaluates model.

    Args:
      x_train: feature matrix for training/CV data
      y_train: label array for training/CV data
      x_test: feature matrix for test data
      y_test: label array for test data
      metric: name of metric to be optimized

    Returns:
        Optimization function set up to take parameter dict from Hyperopt.
    """

    def train_func(params):
        """Train a model and return loss metric."""
        metrics = fit_and_log_cv(
          x_train, y_train, x_test, y_test, params, nested=True)
        return {'status': hyperopt.STATUS_OK, 'loss': metrics[metric]}

    return train_func


def log_best(run: mlflow.entities.Run,
         metric: str) -> None:
    """Log the best parameters from optimization to the parent experiment.

    Args:
        run: current run to log metrics
        metric: name of metric to select best and log
    """

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        [run.info.experiment_id],
        "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id))

    best_run = min(runs, key=lambda run: run.data.metrics[metric])

    mlflow.set_tag("best_run", best_run.info.run_id)
    mlflow.log_metric(f"best_{metric}", best_run.data.metrics[metric])


from hyperopt.pyll.base import scope

MAX_EVALS = 200
METRIC = "val_RMSE"
# Number of experiments to run at once
PARALLELISM = 8

space = {
    'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.5, 1.0),
    'subsample': hyperopt.hp.uniform('subsample', 0.05, 1.0),
    # The parameters below are cast to int using the scope.int() wrapper
    'num_iterations': scope.int(
      hyperopt.hp.quniform('num_iterations', 10, 200, 1)),
    'num_leaves': scope.int(hyperopt.hp.quniform('num_leaves', 20, 50, 1))
}

trials = hyperopt.SparkTrials( parallelism=PARALLELISM )
train_objective = build_train_objective( x_train, y_train, x_test, y_test, METRIC )

with mlflow.start_run() as run:
    hyperopt.fmin(fn=train_objective,
                    space=space,
                    algo=hyperopt.tpe.suggest,
                    max_evals=MAX_EVALS,
                    trials=trials)
    log_best(run, METRIC)
    search_run_id = run.info.run_id
    experiment_id = run.info.experiment_id