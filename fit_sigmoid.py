import matplotlib
matplotlib.use('TkAgg')  # Prevent Qt timer crash on Windows

import matplotlib.pyplot as plt
from matplotlib.image import imread
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import random

# --- CONFIG ---
base_dir = Path("C:/Users/jasco/Desktop/Tesis/Tuning-Reservoir-Computer-based-Tracking-Controller/plots/rc_parallel_test")
n = 10  # Number of images to label
image_ext = ".png"
csv_name = "trial_log.csv"

# --- Gather all PNGs ---
all_pngs = list(base_dir.rglob(f"*{image_ext}"))
random.shuffle(all_pngs)
selected_pngs = all_pngs[:n]

labels = []
features = []
trial_ids = []

# --- Label Loop ---
for path in selected_pngs:
    trial_id = path.stem  # remove .png
    folder = path.parent
    log_path = folder / csv_name

    if not log_path.exists():
        print(f"⚠️ Skipping {trial_id} — no trial_log.csv in {folder}")
        continue

    try:
        df = pd.read_csv(log_path)
        row = df[df["trial_id"] == trial_id]
        if row.empty:
            print(f"⚠️ Skipping {trial_id} — no matching entry in trial_log.csv")
            continue

        rmse = float(row.iloc[0]["rmse"])
    except Exception as e:
        print(f"⚠️ Skipping {trial_id} — failed to extract RMSE: {e}")
        continue

    # Show image
    img = imread(path)
    plt.imshow(img)
    plt.title(f"{trial_id} — RMSE: {rmse:.4f}")
    plt.axis('off')
    plt.show(block=False)

    label = input("Label this as 0 or 1: ").strip()
    plt.close()

    if label not in ['0', '1']:
        print("⚠️ Invalid input, skipping.")
        continue

    features.append(rmse)
    labels.append(int(label))
    trial_ids.append(trial_id)

# --- Prepare data ---
X = np.array(features)
y = np.array(labels)
trial_ids = np.array(trial_ids)

# Remove any NaNs
mask = ~np.isnan(X)
X = X[mask]
y = y[mask]
trial_ids = trial_ids[mask]

if len(X) < 2:
    print("❌ Not enough valid samples to fit sigmoid.")
    exit()

# --- Sigmoid model ---
def sigmoid(x, w, b):
    return expit(w * x + b)

def loss_fn(params, X, y):
    w, b = params
    preds = sigmoid(X, w, b)
    return mean_squared_error(y, preds)

# --- Fit sigmoid ---
result = minimize(loss_fn, x0=[1.0, 0.0], args=(X, y))
w_fit, b_fit = result.x
y_pred = sigmoid(X, w_fit, b_fit)
fit_mse = mean_squared_error(y, y_pred)

# --- Report ---
print("\n=== SIGMOID FIT ===")
print(f"Sigmoid: y = 1 / (1 + exp(-({w_fit:.3f} * rmse + {b_fit:.3f})))")
print(f"Fit MSE: {fit_mse:.5f}")
print(f"Labeled {len(X)} plots")

# --- Save labeled data ---
output_csv = base_dir / "labeled_rmse.csv"
df_out = pd.DataFrame({
    "trial_id": trial_ids,
    "rmse": X,
    "label": y
})
df_out.to_csv(output_csv, index=False)
print(f"✅ Saved labeled data to {output_csv}")
