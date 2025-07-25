import os
import tempfile

import numpy as np
from trial_log_vs_study_validator import validate_logger_vs_study

from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from src.trial_logger import TrialLogger
from src.tuner import Tuner


def build_model(input_shape, output_shape):
    model = RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(RandomReservoirLayer(nodes=200))
    model.add(ReadoutLayer(output_shape=output_shape))
    model.compile(optimizer="ridge", metrics=["mse"])
    return model


def run_tuning_and_validate():
    # === 1. Daten vorbereiten
    omega = np.pi
    t = np.linspace(0, 3 * (2 * np.pi / omega), 300)
    x = np.sin(omega * t)
    y = 2 * np.cos(omega * t)

    x_train = np.expand_dims(x, axis=(0, 2))  # (1, 300, 1)
    y_train = np.expand_dims(y, axis=(0, 2))

    input_shape = x_train.shape[1:]
    output_shape = y_train.shape[1:]

    # === 2. Logger + temporäres Verzeichnis
    tmp_dir = tempfile.mkdtemp()
    csv_path = os.path.join(tmp_dir, "trial_log.csv")
    logger = TrialLogger(log_dir=tmp_dir, csv_name="trial_log.csv")

    # === 3. Hyperparameter-Raum definieren
    search_space = {
        "spec_rad": ("float", 0.6, 1.2),
        "leakage_rate": ("float", 0.1, 1.0),
        "activation": ("categorical", ["tanh", "sigmoid"]),
        "input_scaling": ("float", 0.5, 2.0)
    }

    # === 4. Modell aufbauen
    model = build_model(input_shape, output_shape)

    # === 5. Tuner initialisieren
    tuner = Tuner(
        model=model,
        search_space=search_space,
        x_train=x_train,
        y_train=y_train,
        x_val=x_train,
        y_val=y_train,
        metric="mse",
        n_trials=10,
        trial_logger=logger,
        run_id=0,
        verbose=False
    )

    # === 6. Optuna-Optimierung durchführen
    tuner.optimize()

    # === 7. Vergleich mit trial_log.csv
    validate_logger_vs_study(
        csv_path=csv_path,
        study=tuner.study,
        sampler_type="tpe",
        run_id=0
    )


if __name__ == "__main__":
    run_tuning_and_validate()
