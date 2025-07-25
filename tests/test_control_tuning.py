import scipy.io
from experiment import run_experiment

from src.control import control_loop
from pyreco.utils_data import *
from src.robot import TwoLinkRobot
from src.utils import load_models, load_validation_trajectory

# === Load Robot + Trajectory + Training Data ===
params = scipy.io.loadmat('all_traj_06112025_341_7413.mat', struct_as_record=False, squeeze_me=True)
train_data = scipy.io.loadmat('fixed_robot_training_data.mat', struct_as_record=False, squeeze_me=True)

# Robot
props = params['properties']
m1, m2, l1, l2, lc1, lc2, I1, I2 = props
dt = float(params['dt'])
robot = TwoLinkRobot(m1, m2, l1, l2, lc1, lc2, I1, I2, dt=dt)

# Trajectory
traj_name = "lorenz"
trajectory, q_control, qdt_control, val_length, *_ = load_validation_trajectory(traj_name)

# Training Data
xy = train_data['xy']
qdt = train_data['qdt']
tau = train_data['tau']
N = 1000
X = np.hstack([
    xy[0:N, :], xy[1:N+1, :],
    qdt[0:N, :], qdt[1:N+1, :]
])
Y = tau[0:N, :]
X = X.reshape(-1, 1, 8)
Y = Y.reshape(-1, 1, 2)

# === Control Evaluator ===
def control_loop_evaluator(model, trial):
    model.fit(X, Y)
    pred = control_loop(
        rc=model,
        robot=robot,
        trajectory=trajectory,
        qdt_traj=qdt_control,
        q_init=q_control[0, :],
        qdt_init=qdt_control[0, :],
        disturbance_failure=np.zeros((2, val_length)),
        measurement_failure=np.zeros((4, val_length)),
        taudt_threshold=np.array([-5e-2, 5e-2])
    )
    trial.set_user_attr("y_pred", pred.reshape(1, -1, 2))
    trial.set_user_attr("y_true", trajectory.reshape(1, -1, 2))
    return np.sqrt(np.mean(np.sum((pred.T - trajectory.T) ** 2, axis=0)))

# === Search Space ===
search_space = {
    "spec_rad": ("float", 0.1, 1.0),
    "leakage_rate": ("float", 0.05, 1.0),
    "activation": ("categorical", ["tanh", "sigmoid"]),
    "input_scaling": ("float", 0.1, 1.0),
}

# === Load Saved Reservoir Models ===
models = load_models("saved_reservoirs", n_runs=30)

# === Run Full Experiment with Logging ===
run_experiment(
    models=models,
    methods=["grid", "random", "tpe"],
    n_trials=30,
    X_train=X,
    y_train=Y,
    search_space=search_space,
    experiment_name="rc_control_benchmark",
    evaluator=control_loop_evaluator,
    x_val=X,
    y_val=Y
)
