# 🧠 Hyperparameter Optimization for Reservoir Computing Control

This project benchmarks different **hyperparameter optimization (HPO) methods** for tuning a **Reservoir Computing (RC)** controller used in closed-loop control of a two-link robotic arm tracking desired trajectories.

We compare:
- **Grid Search**
- **Random Search**
- **Bayesian Optimization (TPE)**

Each method is tested over multiple independent runs using fixed random reservoir initializations, and results are saved for later quantitative analysis.

---

## 🚀 What This Project Does

- Runs **many independent hyperparameter tuning experiments** for each method (e.g. 30 runs × 30 trials per method).
- Uses **Optuna** to optimize RC hyperparameters (spectral radius, leakage rate, activation, input scaling).
- Evaluates each trial in **closed-loop trajectory tracking** using a two-link robot model.
- Measures performance using **RMSE** between predicted and ground-truth end-effector trajectories.
- Saves:
  - RMSE and hyperparameters for every trial (CSV)
  - Plots comparing prediction vs. ground truth
  - The best model per run (`.joblib`)
- Enables **offline analysis** of HPO performance across metrics like convergence speed, failure rate, and generalization gap.

---

## 🎯 Why This Matters

Reservoir Computing is fast and lightweight, but highly sensitive to hyperparameters. This project provides a **reproducible framework** for comparing optimization methods and understanding how tuning affects control performance in a real robotic task.

It is designed to:
- Support **statistical evaluation** across many runs
- Enable **fair comparisons** by sharing random seeds/reservoirs across methods
- Assist in **academic studies** of control-oriented hyperparameter tuning

---

## 🧪 Running Experiments

To launch the full benchmark (Grid, Random, TPE):

```bash
python test/test_control_tuning.py
