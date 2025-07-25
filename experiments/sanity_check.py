import os

import joblib
import numpy as np

from pyreco.utils_networks import compute_spec_rad, set_spec_rad


def mse(a, b):
    return np.mean((a - b) ** 2)

def match_reservoirs(best_model_dir, reservoir_dir, num_runs=30):
    print("=== Matching best models to original reservoirs (using spectral radius scaling) ===")

    for run_id in range(num_runs):
        best_model_path = os.path.join(best_model_dir, f"best_model_run_{run_id}.joblib")
        if not os.path.isfile(best_model_path):
            print(f"[Run {run_id:02d}] ❌ File not found: {best_model_path}")
            continue

        best_model = joblib.load(best_model_path)
        W_best = best_model.reservoir_layer.weights
        target_spec_rad = compute_spec_rad(W_best)

        print(f"\n[Run {run_id:02d}] Target spec_rad = {target_spec_rad:.6f}")
        best_match = None
        best_error = float("inf")
        best_original_spec = None

        for i in range(num_runs):
            reservoir_path = os.path.join(reservoir_dir, f"reservoir_{i}.joblib")
            if not os.path.isfile(reservoir_path):
                print(f"  [reservoir_{i:02d}] ❌ Missing file: {reservoir_path}")
                continue

            reservoir = joblib.load(reservoir_path)
            W_orig = reservoir.reservoir_layer.weights.copy()
            orig_spec_rad = compute_spec_rad(W_orig)

            # Rescale original weights to match target spectral radius
            try:
                W_scaled = set_spec_rad(W_orig.copy(), target_spec_rad)
            except Exception as e:
                print(f"  [reservoir_{i:02d}] ⚠️ Scaling failed: {e}")
                continue

            error = mse(W_best, W_scaled)
            print(f"  [reservoir_{i:02d}] MSE: {error:.6e} | Orig spec_rad: {orig_spec_rad:.6f}")

            if error < best_error:
                best_error = error
                best_match = i
                best_original_spec = orig_spec_rad

        print(f"✅ Best match: reservoir_{best_match:02d} | MSE: {best_error:.6e} | Orig spec_rad: {best_original_spec:.6f} → {target_spec_rad:.6f}")

if __name__ == "__main__":
    # === CONFIGURE THIS ===
    best_model_dir = "plots/rc_parallel_test/tpe"       # path to your best_model_run_*.joblib
    reservoir_dir = "saved_reservoirs"                  # path to original reservoir_*.joblib
    num_runs = 30                                       # number of runs

    match_reservoirs(best_model_dir, reservoir_dir, num_runs)
