import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort

CSV = "data/processed/train.csv"
SCALER = "data/models/ae_scaler.joblib"
MODEL = "data/models/ae.omx"
OUT = "data/processed/ae_reconstruction_errors.csv"

df = pd.read_csv(CSV)

X = (
    df.select_dtypes(include=[np.number])
      .drop(columns=["label"], errors="ignore")
      .values.astype(np.float32)
)

scaler = joblib.load(SCALER)
X = scaler.transform(X).astype(np.float32)

sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
recon = sess.run(None, {inp: X})[0]

mse = np.mean((X - recon) ** 2, axis=1)

out_df = pd.DataFrame({
    "reconstruction_error": mse
})

out_df.to_csv(OUT, index=False)
print(f"[OK] Saved reconstruction errors to {OUT}")
print("Mean MSE:", mse.mean())
print("Std  MSE:", mse.std())
