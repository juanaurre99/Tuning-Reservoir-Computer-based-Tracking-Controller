from experiment_paralelle import run_experiment, control_loop_evaluator
from utils import load_models, load_validation_trajectory
from robot import TwoLinkRobot
import scipy.io
import numpy as np
from functools import partial
import json


def main():
    # === Load experiment config ===
    with open("config.json", "r") as f:
        config = json.load(f)
    exp = config["experiment"]

    # === Robot Parameters ===
    r = config["robot_params"]
    robot = TwoLinkRobot(
        r["m1"], r["m2"], r["l1"], r["l2"],
        r["lc1"], r["lc2"], r["I1"], r["I2"],
        dt=r["dt"]
    )

    # === Load Training Data ===
    train_data = scipy.io.loadmat(config["data"]["train_data_file"], struct_as_record=False, squeeze_me=True)
    xy, qdt, tau = train_data['xy'], train_data['qdt'], train_data['tau']

    if exp["training_length"] == "Full":
        N = xy.shape[0] - 1
    else:
        N = exp["training_length"]

    X = np.hstack([xy[0:N], xy[1:N+1], qdt[0:N], qdt[1:N+1]]).reshape(-1, 1, 8)
    Y = tau[0:N].reshape(-1, 1, 2)

    # === Load individual validation trajectories ===
    val_names = config["data"]["validation_trajectories"]
    trajectory_tuples = [load_validation_trajectory(name) for name in val_names]

    # === Evaluator Function ===
    evaluator_fn = partial(
        control_loop_evaluator,
        X=X,
        Y=Y,
        robot=robot,
        trajectories=trajectory_tuples
    )

    # === Run Experiment ===
    run_experiment(
        models=load_models(exp["model_dir"], n_runs=exp["n_runs"]),
        methods=exp["methods"],
        n_trials=exp["n_trials"],
        X_train=X,
        y_train=Y,
        search_space=config["search_space"],
        experiment_name=exp["experiment_name"],
        evaluator=evaluator_fn,
        x_val=X,
        y_val=Y,
        n_processes=exp["n_processes"]
    )


if __name__ == "__main__":
    main()
