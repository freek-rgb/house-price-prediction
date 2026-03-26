"""
House Price Prediction API
--------------------------
Run with:  uvicorn app:app --reload --port 8000
Then open: http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# ── Recreate the exact transformer used in the notebook ──────────────────────
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household      = X[:, rooms_ix]      / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        return np.c_[X, rooms_per_household, population_per_household]


# ── Column order must match training ─────────────────────────────────────────
NUM_ATTRIBS = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income",
]
CAT_ATTRIBS = ["ocean_proximity"]

# All 5 categories the model was trained on — must be hardcoded so
# OneHotEncoder always produces exactly 5 columns regardless of input
ALL_OCEAN_CATEGORIES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

num_pipeline = Pipeline([
    ("imputer",       SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler",    StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, NUM_ATTRIBS),
    ("cat", OneHotEncoder(
        categories=[ALL_OCEAN_CATEGORIES],
        handle_unknown="ignore"
    ), CAT_ATTRIBS),
])

# ── Fit pipeline once on a dummy dataset covering all categories ──────────────
# This ensures the pipeline is always ready and produces exactly 16 features
_dummy = pd.DataFrame([
    {
        "longitude": -122.0, "latitude": 37.0, "housing_median_age": 20,
        "total_rooms": 1000, "total_bedrooms": 200, "population": 500,
        "households": 200, "median_income": 5.0, "ocean_proximity": cat
    }
    for cat in ALL_OCEAN_CATEGORIES
])
full_pipeline.fit(_dummy)
print("Pipeline fitted ✓  →", full_pipeline.transform(_dummy[:1]).shape[1], "features")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading forest_reg.pkl ...")
model = joblib.load("forest_reg.pkl")
print("Model loaded ✓")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="House Price Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_frontend():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    return FileResponse(html_path)

# ── Schema ────────────────────────────────────────────────────────────────────
class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str  # "<1H OCEAN" | "INLAND" | "ISLAND" | "NEAR BAY" | "NEAR OCEAN"


# ── Predict ───────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(features: HouseFeatures):
    df = pd.DataFrame([features.model_dump()])
    prepared = full_pipeline.transform(df)   # pipeline already fitted, just transform
    price = float(model.predict(prepared)[0])
    return {
        "predicted_price":          round(price, 2),
        "rooms_per_household":      round(features.total_rooms    / features.households, 3),
        "bedrooms_per_room":        round(features.total_bedrooms / features.total_rooms, 3),
        "population_per_household": round(features.population     / features.households, 3),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
