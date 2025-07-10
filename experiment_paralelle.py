import os
import joblib
import numpy as np
from typing import List, Tuple
from functools import partial
from multiprocessing import Pool, cpu_count

from tuner import Tuner
from pyreco.trial_logger import TrialLogger
from pyreco.metrics import mse
from control import control_loop


# === Windows-safe TOP-LEVEL evaluator ===
def control_loop_evaluator(
    model, trial, X, Y, robot, trajectory, q_control, qdt_control, val_length
):
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


# === Experiment runner for a single run (inside subprocess) ===
def _run_single_experiment(
    method: str,
    run_id: int,
    model,
    log_dir: str,
    X_train,
    y_train,
    x_val,
    y_val,
    search_space,
    n_trials,
    evaluator,
):
    model_filename = f"best_model_run_{run_id}.joblib"
    model_path = os.path.join(log_dir, model_filename)

    if os.path.exists(model_path):
        print(f"[Resume] Skipping {method.upper()} run {run_id}: model already exists.")
        return

    print(f"\n>>> {method.upper()} Run {run_id} <<<")

    logger = TrialLogger(log_dir=log_dir, csv_name="trial_log.csv")

    tuner = Tuner(
        model=model,
        search_space=search_space,
        x_train=X_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        metric=mse if evaluator is None else None,
        evaluator=evaluator,
        n_trials=n_trials,
        sampler_type=method,
        verbose=True,
        trial_logger=logger,
        run_id=run_id,
    )

    tuner.optimize()
    tuner.report()
    logger.save_model(tuner.best_model, filename=model_filename)


# === Main parallel runner ===
def run_experiment(
    models: List,
    methods: List[str],
    n_trials: int,
    X_train,
    y_train,
    search_space: dict,
    experiment_name: str = "full_experiment",
    evaluator=None,
    x_val=None,
    y_val=None,
    n_processes=None,
):
    assert len(models) > 0, "Model list is empty"
    output_base = os.path.join("plots", experiment_name)
    os.makedirs(output_base, exist_ok=True)
    n_processes = n_processes or min(cpu_count(), len(models))

    for method in methods:
        print(f"\n=== Starting method: {method.upper()} ===")
        log_dir = os.path.join(output_base, method)
        os.makedirs(log_dir, exist_ok=True)

        # Prepare tasks
        tasks: List[Tuple] = []
        for run_id, model in enumerate(models):
            tasks.append((
                method, run_id, model, log_dir,
                X_train, y_train, x_val, y_val,
                search_space, n_trials, evaluator
            ))

        # Parallel execution
        with Pool(processes=n_processes) as pool:
            pool.starmap(_run_single_experiment, tasks)
