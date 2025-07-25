import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io

from src.control import control_loop
from pyreco.custom_models import RC
from src.robot import TwoLinkRobot
from src.utils import func_rmse, load_validation_trajectory

# === Configuration ===
run_id = 1
trial_index = 2
method = "tpe"

reservoir_path = f"./saved_reservoirs/reservoir_{run_id}.joblib"
log_csv = f"./plots/rc_experiment_test/{method}/trial_log.csv"
train_data_path = "./fixed_robot_training_data.mat"
config_path = "./config.json"
plot_dir = os.path.join("plots", "rc_manual_eval", method)
os.makedirs(plot_dir, exist_ok=True)

# === Load robot configuration ===
with open(config_path, "r") as f:
    config = json.load(f)
r = config["robot_params"]

robot = TwoLinkRobot(
    m1=r["m1"], m2=r["m2"],
    l1=r["l1"], l2=r["l2"],
    lc1=r["lc1"], lc2=r["lc2"],
    I1=r["I1"], I2=r["I2"],
    dt=r["dt"]
)

# === Load training data ===
data = scipy.io.loadmat(train_data_path, struct_as_record=False, squeeze_me=True)
xy, qdt, tau = data['xy'], data['qdt'], data['tau']

N = xy.shape[0] - 1
X = np.hstack([xy[0:N], xy[1:N+1], qdt[0:N], qdt[1:N+1]]).reshape(-1, 1, 8)
Y = tau[0:N].reshape(-1, 1, 2)

# === Load reservoir model ===
model: RC = joblib.load(reservoir_path)

# === Load best trial HPs from CSV ===
df = pd.read_csv(log_csv)
row = df[
    (df["method"] == method) &
    (df["run_id"] == run_id) &
    (df["trial_index"] == trial_index)
].iloc[0]
params = json.loads(row["params"])
model.set_hp(**params)

# === Retrain model ===
#model.fit(X, Y, n_init=1)

reservoir_path = f"./plots/rc_experiment_test/tpe/best_model_run_{run_id}.joblib"
model = joblib.load(reservoir_path)
# === Load validation trajectory ===
trajectory, q_control, qdt_control, val_length, x_control, y_control = load_validation_trajectory("circle")

# === Run control loop ===
data_pred = control_loop(
    rc=model,
    robot=robot,
    trajectory=trajectory,
    qdt_traj=qdt_control,
    q_init=q_control[0],
    qdt_init=qdt_control[0],
    disturbance_failure=np.zeros((2, val_length)),
    measurement_failure=np.zeros((4, val_length)),
    taudt_threshold=np.array([-5e-2, 5e-2])
)

# === Compute RMSE using utils.func_rmse
rmse = func_rmse(data_pred, trajectory, 0, val_length - 1)

# === Format for consistent plotting ===
# Convert (2, T) → (T, 2) if necessary
if trajectory.shape[0] == 2:
    trajectory = trajectory.T
if data_pred.shape[0] == 2:
    data_pred = data_pred.T

# Add batch dimension → (1, T, 2)
trajectory = trajectory[np.newaxis, :, :]
data_pred = data_pred[np.newaxis, :, :]

# === Compute RMSE
rmse = func_rmse(data_pred[0], trajectory[0], 0, val_length - 1)

# === Plot and save
plt.figure(figsize=(7, 6))
plt.plot(trajectory[0, :, 0], trajectory[0, :, 1], '--', label='Ground Truth', linewidth=2)
plt.plot(data_pred[0, :, 0], data_pred[0, :, 1], label='RC Prediction', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'{method.upper()} | Run {run_id} Trial {trial_index} | RMSE = {rmse:.5f}')
plt.legend()
plt.axis('equal')
plt.tight_layout()

trial_id = f"{method}_{run_id:02}_{trial_index:02}"
plot_path = os.path.join(plot_dir, f"{trial_id}.png")
plt.show()
plt.close()
print(f"[Plot] Saved to {plot_path}")
