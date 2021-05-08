import pandas as pd
from TaxiFareModel.utils import simple_time_tracker
import numpy as np

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"
LOCAL_PATH = "raw_data/train.csv"
GCP_BUCKET_NAME = "wagon-ml-zastrow-566"
GCP_BUCKET_TRAIN_DATA_PATH = "data/train_1k.csv"


DIST_ARGS = dict(
    start_lat="pickup_latitude",
    start_lon="pickup_longitude",
    end_lat="dropoff_latitude",
    end_lon="dropoff_longitude",
)


@simple_time_tracker
def get_data(nrows=10000, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    data_origin = kwargs["data_origin"]
    if data_origin == "local":
        print(f"-> loading from local folder")
        path = LOCAL_PATH
        df = pd.read_csv(path, nrows=nrows)
    elif data_origin == "aws":
        print(f"-> loading from aws")
        path = AWS_BUCKET_PATH
        df = pd.read_csv(path, nrows=nrows)
    elif data_origin == "gcp":
        print(f"-> loading from gcp")
        path = f"gs://{GCP_BUCKET_NAME}/{GCP_BUCKET_TRAIN_DATA_PATH}"
        df = pd.read_csv(path, nrows=nrows)
    return df


def clean_df(df, test=False):
    df = df.dropna(how="any", axis="rows")
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

def df_optimized(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numeircal columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return: df optimized
    """
    in_size = df.memory_usage(index=True).sum()
    # Optimized size here
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df

if __name__ == "__main__":
    params = dict(
        nrows=1000,
        local=True,  # set to False to get data from GCP (Storage or BigQuery)
    )
    df = get_data(**params)
