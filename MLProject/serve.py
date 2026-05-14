import joblib
import uvicorn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()
model = joblib.load('saved_model/model.pkl')

class PredictRequest(BaseModel):
    features: List[float]

@app.post('/invocations')
def predict(req: PredictRequest):
    X = np.array(req.features).reshape(1, -1)
    pred = model.predict(X)[0]
    return {'predictions': [float(pred)]}

@app.get('/ping')
def ping():
    return {'status': 'healthy'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)
