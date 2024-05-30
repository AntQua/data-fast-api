import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pytz import timezone
import numpy as np
from taxifare.ml_logic.registry import load_model
from taxifare.ml_logic.preprocessor import preprocess_features

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
@app.get("/predict")
def predict(
        pickup_datetime: str,  # 2014-07-06 19:18:00
        pickup_longitude: float,    # -73.950655
        pickup_latitude: float,     # 40.783282
        dropoff_longitude: float,   # -73.984365
        dropoff_latitude: float,    # 40.769802
        passenger_count: int
    ):      # 1
    """
    Make a single course prediction.
    Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
    Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
    """


    # model = initialize_model((6,))
    # compile_model(model, learning_rate=0.0005)
    # train_model(
    #     model,
    #     X: np.ndarray,
    #     y: np.ndarray,
    #     batch_size=256,
    #     patience=2,
    #     validation_data=None, # overrides validation_split
    #     validation_split=0.3
    # )[0]

    # prediction = model.predict()

        # Convert pickup_datetime to the correct format and timezone
    try:
        eastern = timezone('US/Eastern')
        pickup_datetime = eastern.localize(datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S"))
        pickup_datetime_utc = pickup_datetime.astimezone(timezone('UTC'))
    except Exception as e:
        return {"error": f"Invalid date format: {str(e)}"}

    # Create a DataFrame with the input data
    data = pd.DataFrame([{
        'pickup_datetime': pickup_datetime_utc,
        'pickup_longitude': pickup_longitude,
        'pickup_latitude': pickup_latitude,
        'dropoff_longitude': dropoff_longitude,
        'dropoff_latitude': dropoff_latitude,
        'passenger_count': passenger_count
    }])

    # Preprocess the data
    X_pred = preprocess_features(data)

    # Load the model
    model = load_model()

    # Make the prediction
    prediction = model.predict(X_pred)[0][0]

    # Convert prediction to a native Python float
    prediction = float(prediction)

    # Return the result as JSON
    return {'fare': prediction}


@app.get("/")
def root():
    return {'greeting': 'Hello'}
