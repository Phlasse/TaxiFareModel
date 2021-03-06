import pandas as pd
import pygeohash as gh
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized, minkowski_distance
from TaxiFareModel.data import get_data, clean_df, DIST_ARGS, df_optimized


class DistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, distance_type="euclidian", **kwargs):
        self.distance_type = distance_type

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        if self.distance_type == "haversine":
            X["distance"] = haversine_vectorized(X, **DIST_ARGS)
        if self.distance_type == "euclidian":
            X["distance"] = minkowski_distance(X, p=2, **DIST_ARGS)
        if self.distance_type == "manhattan":
            X["distance"] = minkowski_distance(X, p=1, **DIST_ARGS)
        return X[["distance"]]

    def fit(self, X, y=None):
        return self

class DataframeCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def transform(self, X, y=None):
        X = pd.DataFrame(X.toarray())
        assert isinstance(X, pd.DataFrame)
        X = df_optimized(X)
        if self.verbose:
            print(X.head())
        return X
    
    def fit(self, X, y=None):
        return self

class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, time_column, time_zone_name="America/New_York"):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_column])
        X_.index = X_.index.tz_convert(self.time_zone_name)
        X_["dow"] = X_.index.weekday
        X_["hour"] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[["dow", "hour", "month", "year"]]

    def fit(self, X, y=None):
        return self


class AddGeohash(BaseEstimator, TransformerMixin):
    def __init__(self, precision=6):
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("geo")
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_["geohash_pickup"] = X_.apply(
            lambda x: gh.encode(
                x.pickup_latitude, x.pickup_longitude, precision=self.precision
            ),
            axis=1,
        )
        print("geo inbetween")
        X_["geohash_dropoff"] = X_.apply(
            lambda x: gh.encode(
                x.dropoff_latitude, x.dropoff_longitude, precision=self.precision
            ),
            axis=1,
        )
        print(X_.head(1))
        return X_[["geohash_pickup", "geohash_dropoff"]]


class DistanceToCenter(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def transform(self, X, y=None):
        X_ = X.copy()
        nyc_center = (40.7141667, -74.0063889)
        X_["nyc_lat"], X_["nyc_lng"] = nyc_center[0], nyc_center[1]
        args_pickup = dict(
            start_lat="nyc_lat",
            start_lon="nyc_lng",
            end_lat="pickup_latitude",
            end_lon="pickup_longitude",
        )
        args_dropoff = dict(
            start_lat="nyc_lat",
            start_lon="nyc_lng",
            end_lat="dropoff_latitude",
            end_lon="dropoff_longitude",
        )
        X_["pickup_distance_to_center"] = haversine_vectorized(X_, **args_pickup)
        X_["dropoff_distance_to_center"] = haversine_vectorized(X_, **args_dropoff)
        return X_[["pickup_distance_to_center", "dropoff_distance_to_center"]]

    def fit(self, X, y=None):
        return self


class Direction(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        start_lat="pickup_latitude",
        start_lon="pickup_longitude",
        end_lat="dropoff_latitude",
        end_lon="dropoff_longitude",
    ):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def transform(self, X, y=None):
        X_ = X.copy()

        def calculate_direction(d_lon, d_lat):
            result = np.zeros(len(d_lon))
            l = np.sqrt(d_lon ** 2 + d_lat ** 2)
            result[d_lon > 0] = (180 / np.pi) * np.arcsin(
                d_lat[d_lon > 0] / l[d_lon > 0]
            )
            idx = (d_lon < 0) & (d_lat > 0)
            result[idx] = 180 - (180 / np.pi) * np.arcsin(d_lat[idx] / l[idx])
            idx = (d_lon < 0) & (d_lat < 0)
            result[idx] = -180 - (180 / np.pi) * np.arcsin(d_lat[idx] / l[idx])
            return result

        X_["delta_lon"] = X_[self.start_lon] - X_[self.end_lon]
        X_["delta_lat"] = X_[self.start_lat] - X_[self.end_lat]
        X_["direction"] = calculate_direction(X_.delta_lon, X_.delta_lat)
        return X_[["delta_lon", "delta_lat", "direction"]]

    def fit(self, X, y=None):
        return self


if __name__ == "__main__":
    params = dict(
        nrows=1000,
        upload=False,
        local=False,  # set to False to get data from GCP (Storage or BigQuery)
        optimize=False,
    )
    df = get_data(**params)
    df = clean_df(df)
    dir = Direction()
    dist_to_center = DistanceToCenter()
    X = dir.transform(df)
    X2 = dist_to_center.transform(df)
