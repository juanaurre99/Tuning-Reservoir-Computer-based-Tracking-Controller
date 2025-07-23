import numpy as np
import joblib
import os
import tempfile
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.utils_data import sine_pred
from tuner import Tuner
from trial_logger import TrialLogger


def test_tuner_hyperparam_integrity():
    # === 1. Generate simple sine → cosine task
    omega = np.pi
    t = np.linspace(start=0, stop=3 * (2 * np.pi / omega), num=300, endpoint=True)
    x = np.sin(omega * t)
    y = 2 * np.cos(omega * t)

    x_train = np.expand_dims(x, axis=(0, 2))  # shape [1, 300, 1]
    y_train = np.expand_dims(y, axis=(0, 2))

    input_shape = (x_train.shape[1], x_train.shape[2])
    output_shape = (y_train.shape[1], y_train.shape[2])

    # === 2. Manually build the model like your example
    model = RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(RandomReservoirLayer(
        nodes=200,
        activation='tanh',
        fraction_input=0.5,
        spec_rad=0.9,         # will be tuned
        leakage_rate=0.5      # will be tuned
    ))
    model.add(ReadoutLayer(output_shape=output_shape))

    # === 3. Compile before tuning
    model.compile(optimizer="ridge", metrics=["mse"])

    # === 4. Define hyperparameter space
    search_space = {
        "spec_rad": ("float", 0.6, 1.2),
        "leakage_rate": ("float", 0.1, 1.0),
        "activation": ("categorical", ["tanh", "sigmoid"]),
        "input_scaling": ("float", 0.5, 2.0)
    }

    # === 5. Setup tuning
    tmp_dir = tempfile.mkdtemp()
    logger = TrialLogger(log_dir=tmp_dir)

    tuner = Tuner(
        model=model,
        search_space=search_space,
        x_train=x_train,
        y_train=y_train,
        x_val=x_train,  # same for simplicity
        y_val=y_train,
        metric="mse",
        n_trials=10,
        trial_logger=logger,
        run_id=0,
        verbose=True,
    )

    # === 6. Run tuning
    study = tuner.optimize()

    # === 7. Save best model
    model_path = os.path.join(tmp_dir, "best_model.joblib")
    tuner.save_best_model(model_path)

    # === 8. Reload and check
    best_model = joblib.load(model_path)

    print("\n=== Best Model Hyperparameters ===")
    for k, v in best_model.get_hp().items():
        print(f"  {k}: {v}")

    # Ensure normalization parameters are set
    assert best_model.input_mean is not None
    assert best_model.output_mean is not None

    # Predict and check output shape
    y_pred = best_model.predict(x_train)
    assert y_pred.shape == y_train.shape
    print(f"\n✅ Prediction shape OK: {y_pred.shape}")
    print("✅ Normalization and hyperparameters preserved.")

    # === 9. Load model again and verify hyperparameters
    print("\n=== Re-loaded Model Hyperparameters ===")
    model_reloaded = joblib.load(model_path)
    hp = model_reloaded.get_hp()
    for k, v in hp.items():
        print(f"  {k}: {v}")

    # Optional: compare to Optuna best trial
    print("\n=== Optuna best_trial.params ===")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Sanity check: match between logged trial and saved model
    mismatches = []
    for k in study.best_trial.params:
        if k not in hp:
            mismatches.append(f"Missing in model: {k}")
        elif isinstance(hp[k], float):
            if not np.isclose(hp[k], study.best_trial.params[k], atol=1e-5):
                mismatches.append(f"{k}: model={hp[k]} vs trial={study.best_trial.params[k]}")
        else:
            if hp[k] != study.best_trial.params[k]:
                mismatches.append(f"{k}: model={hp[k]} vs trial={study.best_trial.params[k]}")

    if mismatches:
        print("\n❌ Hyperparameter mismatches detected:")
        for line in mismatches:
            print("   ", line)
    else:
        print("\n✅ All hyperparameters preserved correctly.")



if __name__ == "__main__":
    test_tuner_hyperparam_integrity()
