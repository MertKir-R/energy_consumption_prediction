import json
from pathlib import Path
import pandas as pd
import joblib

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

HERE = Path(__file__).resolve().parent          # .../src
ROOT = HERE.parent                              # project root

RUN_DIR  = ROOT / "runs" / "xgb_run_01"

MODEL_PATH = RUN_DIR / "model.joblib" # from train.py
COLS_PATH  = RUN_DIR / "train_columns.json" # from train.py

# 1. Initialize the app
app = FastAPI(title="Energy Consumption Prediction API")

# Load the model at startup
model = joblib.load(MODEL_PATH)

# Load the columns at startup
with open(COLS_PATH, "r") as f:
    train_columns = json.load(f)


# fastAPI uses pydantic which enforces and validates the types and variables we passed
# inside the functions under the endpoints like post, get etc will be 
# casted to integer and if it is not possible it will raise valuerror. To avoid 
# writing each columns' type inside functions we create a class and pass that to functions
class EnergyInput(BaseModel):
    year: int
    population: float
    gdp_wb: float
    energy_per_capita_lag1: float
    solar_consumption_lag1: float
    wind_consumption_lag1: float
    hydro_consumption_lag1: float
    biofuel_consumption_lag1: float
    other_renewable_consumption_lag1: float
    nuclear_consumption_lag1: float
    coal_production_lag1: float
    gas_production_lag1: float
    oil_electricity_lag1: float
    oil_production_lag1: float
    oil_share_elec_lag1: float
    oil_share_energy_lag1: float
    energy_per_gdp_new_lag1: float
    oil_prod_per_capita_new_lag1: float
    oil_consumption_lag1: float
    oil_consumption_lag3: float
    oil_consumption_lag5: float
    gas_consumption_lag1: float
    gas_consumption_lag3: float
    gas_consumption_lag5: float
    coal_consumption_lag1: float
    coal_consumption_lag3: float
    coal_consumption_lag5: float
    gdp_wb_avg_lag3: float
    gdp_wb_avg_lag5: float
    gdp_wb_avg_lag7: float
    population_avg_lag3: float
    population_avg_lag5: float
    population_avg_lag7: float
    oil_consumption_avg_lag3: float
    oil_consumption_avg_lag5: float
    oil_consumption_avg_lag7: float
    gas_consumption_avg_lag3: float
    gas_consumption_avg_lag5: float
    gas_consumption_avg_lag7: float
    coal_consumption_avg_lag3: float
    coal_consumption_avg_lag5: float
    coal_consumption_avg_lag7: float
    country: str


@app.post("/predict")
def predict(data: EnergyInput):
    try:
        # When the user clicks "Predict" on a frontend website, the data travels 
        # across the internet as a JSON string
        # Example: {"temperature": 25.5, "country": "Belgium"}
        # When that JSON hits your @app.post("/predict") endpoint, FastAPI hands it to
        # Pydantic. Pydantic reads the JSON, validates it (makes sure 25.5 is a float),
        # and packs it into your EnergyInput class
        # Pandas (pd.DataFrame) doesn't know what an EnergyInput Pydantic object is. 
        # If you try to hand it to Pandas directly, Pandas will throw an error. 
        # Pandas prefers plain, standard Python dictionaries
        # data.model_dump() is a built-in Pydantic command that says: 
        # "Take this strict Pydantic object, strip away the security guards, and dump
        # the raw data into a standard Python dictionary."
        input_dict = data.model_dump()

        country_list = ['Austria', 'Belgium', 'Bulgaria', 'Cyprus',
       'Czechia', 'Denmark', 'Finland',
       'France', 'Germany', 'Greece',
       'Hungary', 'Ireland', 'Italy',
       'Luxembourg', 'Netherlands', 'Poland',
       'Portugal', 'Romania', 'Slovakia',
       'Spain', 'Sweden']

        user_country = input_dict.pop("country").strip().title()
        if user_country not in country_list:
            raise ValueError(f"Invalid country: '{user_country}'. Must be one of: {country_list}")
        # Convert to Pandas DataFrame
        df = pd.DataFrame([input_dict])

        # this way we set country_Belgium = True and thanks to reindex fill_value = False
        # all the other dummy country variables like country_Spain = False
        # user wont have to manually input country_Belgium True and all other as False
        dummy_col_name = f"country_{user_country}"
        df[dummy_col_name] = True
        
        df = df.reindex(columns=train_columns, fill_value=False)
        
        # Make the prediction
        prediction = model.predict(df)
        
        # Return the result
        return {"predicted_oil_consumption": float(prediction[0])}
        
    # if anything goes wrong Python jumps down to this except block instead of crashing the program
    # e is the actual Python error message stored as a variable    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Standard Python errors (ValueError, KeyError) mean nothing to the internet. 
    # HTTPException translates the Python crash into an official internet error that
    # browsers and web apps understand. 404 means "Not Found" 400 means "Bad Request"
    # (str(e)) will give you the exact reason it failed (e.g., "ValueError: Cannot convert string to float")

@app.get("/health")
def health_check():
    return {"status": "API is running"}