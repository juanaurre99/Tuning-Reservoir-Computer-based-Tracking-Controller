import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional

from pyreco.custom_models import RC
from pyreco.tuner import Tuner
from pyreco.trial_logger import TrialLogger


class Experiment:
    def __init__(
        self,
        name: str,
        method: str,                              # 'grid', 'random', 'bayes'
        run_id: int,
        model: RC,                                # Pre-initialized RC model with fixed weights
        tuner: Tuner,                             # Initialized Tuner (search space, etc.)
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        trial_logger: Optional[TrialLogger] = None,
        save_dir: str = "models"
    ):
        self.name = name
        self.method = method
        self.run_id = run_id
        self.model = model
        self.tuner = tuner
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val if x_val is not None else x_train
        self.y_val = y_val if y_val is not None else y_train
        self.trial_logger = trial_logger
        self.save_dir = save_dir

        self.results = {}

    def run(self, n_runs: int = 1):
        for run in range(n_runs):
            self.run_id = run
            print(f"\n=== {self.method.upper()} | Run {self.run_id} ===")

            self.tuner.model = self.model
            study = self.tuner.optimize()

            best_score = study.best_trial.value
            best_params = study.best_trial.params
            best_trial_index = study.best_trial.number

            # Logging all trials
            best_so_far = float('inf')
            for trial in study.trials:
                if trial.value is None:
                    continue

                trial_id = f"{self.method}_{self.run_id:02}_{trial.number:02}"
                rmse = trial.value
                params = trial.params
                best_so_far = min(best_so_far, rmse)

                if self.trial_logger:
                    y_pred = self.model.predict(self.x_val)
                    self.trial_logger.log(
                        trial_id=trial_id,
                        method=self.method,
                        run_id=self.run_id,
                        trial_index=trial.number,
                        rmse=rmse,
                        best_so_far=best_so_far,
                        y_true=self.y_val,
                        y_pred=y_pred,
                        params=params
                    )

            # Save tuned model and config
            self._save_model(self.model, best_params)

            # Save final results
            self.results = {
                "method": self.method,
                "run_id": self.run_id,
                "best_score": best_score,
                "best_params": best_params,
                "best_trial_index": best_trial_index,
                "convergence_curve": self._compute_convergence_curve(study),
                "y_pred": self.model.predict(self.x_val),
                "y_true": self.y_val,
            }

            self.report()
            self.plot_convergence()
            self.plot_predictions()

    def _compute_convergence_curve(self, study):
        rmses = [t.value for t in study.trials if t.value is not None]
        best_so_far = []
        best = float("inf")
        for r in rmses:
            best = min(best, r)
            best_so_far.append(best)
        return best_so_far

    def _save_model(self, model, best_params):
        os.makedirs(self.save_dir, exist_ok=True)
        prefix = os.path.join(self.save_dir, f"{self.method}_run_{self.run_id:02}")
        # Save model
        with open(prefix + ".pkl", "wb") as f:
            pickle.dump(model, f)
        # Save best hyperparameters
        with open(prefix + "_config.json", "w") as f:
            json.dump(best_params, f, indent=2)

    def report(self):
        print(f"\n=== Experiment Report: {self.name} ===")
        print(f"Method: {self.method} | Run: {self.run_id}")
        print(f"Best RMSE: {self.results['best_score']:.5f}")
        print(f"Best Hyperparameters: {self.results['best_params']}")

    def plot_predictions(self):
        y_true = self.results["y_true"][0, :, 0]
        y_pred = self.results["y_pred"][0, :, 0]

        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Ground Truth", linewidth=2)
        plt.plot(y_pred, label="Tuned Prediction", linestyle="--")
        plt.title(f"{self.name} | Prediction vs Ground Truth")
        plt.xlabel("Time step")
        plt.ylabel("Output")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_convergence(self):
        curve = self.results["convergence_curve"]
        plt.figure(figsize=(6, 4))
        plt.plot(curve, marker='o')
        plt.title(f"{self.name} | Convergence Curve")
        plt.xlabel("Trial")
        plt.ylabel("Best-so-far RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
