from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import joblib
import pathlib
import pandas as pd
import numpy as np
from pydantic import BaseModel

parent_path = pathlib.Path(__file__).parent
model_path = f"{parent_path}/car_price_model/car_price_model.pkl"
preprocessor_path = f"{parent_path}/car_price_model/preprocessor.pkl"
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

class carinput(BaseModel):
    Make: str
    Type: str
    Region: str
    Gear_Type: int
    Origin: str
    Options: str
    Engine_Size: float
    Mileage: int
    Negotiable: int
    CarAge: int

app = FastAPI()

url_list = [
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5500"
]

app.add_middleware(
  CORSMiddleware,
  allow_origins = url_list,
  allow_methods = ["*"],
  allow_headers = ["*"]
)


@app.post("/predict")
def prediction(input_data: carinput):
    input_data = pd.DataFrame([input_data.model_dump()])
    transformed_data = preprocessor.transform(input_data)
    result = model.predict(transformed_data)
    return round(sum((np.exp(result)).tolist()))

app.mount("/", StaticFiles(directory=f"{parent_path.parent}/frontend", html=True), name="frontend")