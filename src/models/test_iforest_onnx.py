import numpy as np
import pandas as pd
import onnxruntime as ort

CSV = "data/processed/train.csv"
MODEL = "data/models/iforest.onnx"
LABEL_COL = "label"
THR = -0.200629  # from training summary

df = pd.read_csv(CSV)
X = df.drop(columns=[LABEL_COL]).select_dtypes(include=[np.number]).values

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

scores = sess.run([output_name], {input_name: X.astype(np.float32)})[0].ravel()
anomaly_scores = -scores
y_pred = (anomaly_scores >= THR).astype(int)

print("Predictions shape:", y_pred.shape)
print("First 10 preds:", y_pred[:10])
