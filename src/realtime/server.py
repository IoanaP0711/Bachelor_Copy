from fastapi import FastAPI
import numpy as np
import onnxruntime as ort

app = FastAPI(title="AI NIDS Inference API")
session = ort.InferenceSession("models/nids_model.onnx")

@app.get("/")
def root():
    return {"status": "AI NIDS online"}

@app.post("/predict")
def predict(features: list[float]):
    arr = np.array(features, dtype=np.float32).reshape(1, -1)
    output = session.run(None, {session.get_inputs()[0].name: arr})
    return {"prediction": output[0][0].tolist()}
