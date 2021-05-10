from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from TaxiFareModel.predict import download_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict_fare/")
def create_fare(
    key,
    pickup_datetime,
    pickup_longitude,
    pickup_latitude,
    dropoff_longitude,
    dropoff_latitude,
    passenger_count
    ):
    
    # key = "2013-07-06 17:18:00.000000119"
    # pickup_datetime = "2013-07-06 17:18:00 UTC"
    # pickup_longitude = "-73.950655"
    # pickup_latitude = "40.783282"
    # dropoff_longitude = "-73.984365"
    # dropoff_latitude = "40.769802"
    # passenger_count = "1"

    # build X ⚠️ beware to the order of the parameters ⚠️
    
    
    X = pd.DataFrame(dict(
        key=[key],
        pickup_datetime=[pickup_datetime],
        pickup_longitude=[float(pickup_longitude)],
        pickup_latitude=[float(pickup_latitude)],
        dropoff_longitude=[float(dropoff_longitude)],
        dropoff_latitude=[float(dropoff_latitude)],
        passenger_count=[int(passenger_count)]))
    
    pipline = download_model()
    results = pipline.predict(X)
    pred = float(results[0])
    return dict(
        prediction=pred)