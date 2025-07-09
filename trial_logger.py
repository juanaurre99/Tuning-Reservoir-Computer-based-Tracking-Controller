import os
import csv
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np


class TrialLogger:
    def __init__(self, log_dir="plots/manual_labeling", csv_name="trial_log.csv"):
        self.log_dir = log_dir
        self.csv_path = os.path.join(self.log_dir, csv_name)

        os.makedirs(self.log_dir, exist_ok=True)

        # Write CSV header if the file doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trial_id",
                    "method",
                    "run_id",
                    "trial_index",
                    "rmse",
                    "best_so_far_rmse",
                    "params"
                ])

    def log(
        self,
        trial_id: str,
        method: str,
        run_id: int,
        trial_index: int,
        rmse: float,
        best_so_far: float,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        params: dict,
    ):
        """Log and plot a trial result with convergence info and hyperparameters."""
        self._plot_prediction(y_true, y_pred, trial_id, rmse, method, run_id, trial_index)

        # Append metadata to CSV
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trial_id,
                method,
                run_id,
                trial_index,
                rmse,
                best_so_far,
                json.dumps(params)
            ])

    def _plot_prediction(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        trial_id: str,
        rmse: float,
        method: str,
        run_id: int,
        trial_index: int,
    ):
        """Plot x-y trajectory and save to disk with debug info."""
        print(f"[Plot] Plotting trial {trial_id}")
        print(f"[Plot] y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

        try:
            plt.figure(figsize=(7, 6))
            plt.plot(y_true[0, :, 0], y_true[0, :, 1], label='Ground Truth', linewidth=2)
            plt.plot(y_pred[0, :, 0], y_pred[0, :, 1], '--', label='RC Prediction', linewidth=2)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'{method.upper()} | Run {run_id} Trial {trial_index} | RMSE = {rmse:.5f}')
            plt.legend()
            plt.axis('equal')
            plt.tight_layout()

            plot_path = os.path.join(self.log_dir, f"{trial_id}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"[Plot] Saved to {plot_path}")
        except Exception as e:
            print(f"[Plot ERROR] Failed to plot trial {trial_id}: {e}")

    def save_model(self, model, filename="best_model.joblib"):
        """Save model using joblib to the logger's directory."""
        path = os.path.join(self.log_dir, filename)
        try:
            joblib.dump(model, path)
            print(f"[TrialLogger] Saved model to {path}")
        except Exception as e:
            print(f"[TrialLogger ERROR] Failed to save model: {e}")
