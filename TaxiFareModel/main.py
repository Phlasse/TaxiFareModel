from TaxiFareModel.data import get_data, clean_df, df_optimized
from TaxiFareModel.predict import generate_submission_csv
from TaxiFareModel.trainer import Trainer
import warnings
from termcolor import colored

warnings.simplefilter(action="ignore", category=FutureWarning)

####################
# Config 4 run
####################
params = dict(
    nrows=50,  # number of samples
    data_origin="gcp",  # Define the origin of the data "local", 'gcp', 'aws'
    is_4_kaggle=False,  # enable kaggle submit
    experiment="[Fed-up!]-Phi-TaxiFare",  # define experiment name for mlflo tracking
    #local=False,  # set to False to get data from aws
    optimize=True,
    estimator="xgboost",
    mlflow=True,  # set to True to log params to mlflow
    experiment_name="[Fed-up!]-Phi-TaxiFare",
    pipeline_memory=None,
    model_upload=True,  # for automatic upload to gcp
    distance_type="manhattan",
    feateng=["distance_to_center", "direction", "distance", "time_features", "geohash"],
)

####################
# Get and clean data
####################
if __name__ == "__main__":
    print("############   Loading Data   ############")
    df = get_data(**params)
    df = clean_df(df)
    df = df_optimized(df)
    y_train = df["fare_amount"]
    X_train = df.drop("fare_amount", axis=1)
    del df
    print("shape: {}".format(X_train.shape))
    print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))

    ####################
    # single model kaggle transmission
    ####################

    if params["is_4_kaggle"] == True:
        print("Auto-Kaggle-submit is challenge is active")
        t = Trainer(X=X_train, y=y_train, **params)
        del X_train, y_train
        t.train()
        t.evaluate()
        t.save_model()
        generate_submission_csv()

    ####################
    # set trainer
    ####################
    else:
        estimators = [
            "GBM"
        ]  # ,"RandomForestRegressor", "Lasso", "Ridge", "LinearRegression", "xgboost", "SGDRegressor"]
        dists = ["haversine"]  # , "manhattan", "euclidian"]
        for estimator in estimators:
            for disti in dists:
                params["estimator"] = estimator
                params["distance_type"] = disti
                print(
                    colored(
                        f"############ {estimator} & {disti}  ############", "yellow"
                    )
                )
                # Train and save model, locally and
                t = Trainer(X=X_train, y=y_train, **params)
                print(colored("############  Training model   ############", "red"))
                t.train()
                print(colored("############  Evaluating model ############", "blue"))
                t.evaluate()
                print(colored("############   Saving model    ############", "green"))
                t.save_model()
