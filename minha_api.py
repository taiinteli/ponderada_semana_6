# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("minha_api")

# Create input/output pydantic models
class InputModel(BaseModel):
    km: float
    bicicleta: float
    caminhao: float
    moto: float
    onibus: float
    outros: float
    utilitarios: float
    BA: float
    CW: float
    DF: float
    ES: float
    GO: float
    MG: float
    MS: float
    MT: float
    PA: float
    PR: float
    RJ: float
    RS: float
    SC: float
    SP: float
    mes: float
    dia: float

class OutputModel(BaseModel):
    prediction: float

# Define predict function
@app.post("/predict", response_model=OutputModel)
def predict(data: InputModel):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["Label"].iloc[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
