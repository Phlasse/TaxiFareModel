import time
import warnings
from TaxiFareModel.data import get_data, clean_df
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from TaxiFareModel.utils import compute_rmse, simple_time_tracker

warnings.filterwarnings("ignore", category=FutureWarning)


class Trainer(object):

    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.kwargs = kwargs
        self.split = self.kwargs.get("split", True)  

    def get_estimator(self):
        """Implement here"""
        estimator = self.kwargs.get("estimator", "RandomForest")
        if estimator == "RandomForest":
            model = RandomForestRegressor()
        return model

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, [
                "pickup_latitude",
                "pickup_longitude",
                'dropoff_latitude',
                'dropoff_longitude'
            ]),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        
        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return round(rmse, 2)


if __name__ == "__main__":
    # Get and clean data
    N = 10_000
    df = get_data(nrows=N)
    df = clean_df(df)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    trainer = Trainer(X_train, y_train)
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
