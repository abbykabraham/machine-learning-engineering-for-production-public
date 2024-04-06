# Import necessary libraries
import pickle
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel, conlist


# Initialize the FastAPI app
app = FastAPI(title="Predicting Wine Class with batching")

# Load the pre-trained classifier from a file in the global scope
# This is done outside of request handling function to load the model only once at the start
with open("models/wine-95.pkl", "rb") as file:
    clf = pickle.load(file)

# Define a Pydantic model for the request body
class Wine(BaseModel):
    batches: List[conlist(item_type=float, min_length=13, max_length=13)]

# Define a route for predictions using a POST request
@app.post("/predict")
def predict(wine: Wine):
    batches = wine.batches
    np_batches = np.array(batches)
    pred = clf.predict(np_batches).tolist()
    return {"Prediction": pred}