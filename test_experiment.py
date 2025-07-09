from experiment_paralelle import run_experiment, control_loop_evaluator
from utils import load_models, load_validation_trajectory
from robot import TwoLinkRobot
import scipy.io
import numpy as np
from functools import partial


def main():
    # === Load data ===
    params = scipy.io.loadmat("all_traj_06112025_341_7413.mat", struct_as_record=False, squeeze_me=True)
    train_data = scipy.io.loadmat("fixed_robot_training_data.mat", struct_as_record=False, squeeze_me=True)

    m1, m2, l1, l2, lc1, lc2, I1, I2 = params['properties']
    dt = float(params['dt'])
    robot = TwoLinkRobot(m1, m2, l1, l2, lc1, lc2, I1, I2, dt=dt)

    trajectory, q_control, qdt_control, val_length, *_ = load_validation_trajectory("lorenz")
    xy, qdt, tau = train_data['xy'], train_data['qdt'], train_data['tau']

    N = xy.shape[0] - 1
    N = 1000
    X = np.hstack([xy[0:N], xy[1:N+1], qdt[0:N], qdt[1:N+1]]).reshape(-1, 1, 8)
    Y = tau[0:N].reshape(-1, 1, 2)

    # === Define search space ===
    search_space = {
        "spec_rad": ("float", 0.1, 1.0),
        "leakage_rate": ("float", 0.05, 1.0),
        "activation": ("categorical", ["tanh", "sigmoid"]),
        "input_scaling": ("float", 0.1, 1.0),
    }

    # === Partial evaluator ===
    evaluator_fn = partial(
        control_loop_evaluator,
        X=X,
        Y=Y,
        robot=robot,
        trajectory=trajectory,
        q_control=q_control,
        qdt_control=qdt_control,
        val_length=val_length,
    )

    # === Run Experiment ===
    run_experiment(
        models=load_models("saved_reservoirs", n_runs=2),
        methods=["grid"],
        n_trials=3,
        X_train=X,
        y_train=Y,
        search_space=search_space,
        experiment_name="rc_parallel_test",
        evaluator=evaluator_fn,
        x_val=X,
        y_val=Y,
        n_processes=2
    )


if __name__ == "__main__":
    main()
