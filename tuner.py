import optuna
import numpy as np
import json
import joblib
import copy
from typing import Optional
from trial_logger import TrialLogger
from pyreco.metrics import assign_metric


class Tuner:
    def __init__(
        self,
        model,
        search_space,
        x_train, y_train,
        x_val=None, y_val=None,
        metric=None,
        n_trials=50,
        sampler_type='tpe',
        num_float_steps=5,
        evaluator=None,
        direction='minimize',
        verbose=False,
        trial_logger: Optional[TrialLogger] = None,
        run_id: int = 0,
    ):
        self.model = model
        self.search_space = search_space
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.metric_name = metric
        self.metric = assign_metric(metric) if metric else None
        self.n_trials = n_trials
        self.sampler_type = sampler_type
        self.num_float_steps = num_float_steps
        self.evaluator = evaluator
        self.direction = direction
        self.verbose = verbose
        self.study = None
        self.trial_logger = trial_logger
        self.run_id = run_id
        self.best_model = None
        self.best_score = float("inf")

    def set_model(self, model):
        self.model = model
        self.best_model = None
        self.best_score = float("inf")
        self.study = None

    def make_grid(self):
        grid = {}
        for k, v in self.search_space.items():
            if v[0] == "categorical":
                grid[k] = v[1]
            elif v[0] == "int":
                grid[k] = list(range(v[1], v[2] + 1))
            elif v[0] == "float":
                grid[k] = list(np.linspace(v[1], v[2], num=self.num_float_steps))
        return grid

    def get_sampler(self):
        if self.sampler_type == 'tpe':
            return optuna.samplers.TPESampler()
        elif self.sampler_type == 'random':
            return optuna.samplers.RandomSampler()
        elif self.sampler_type == 'grid':
            grid = self.make_grid()
            return optuna.samplers.GridSampler(grid)
        else:
            raise ValueError(f"Unknown sampler_type: {self.sampler_type}")

    def suggest_parameters(self, trial):
        params = {}
        for name, (ptype, *args) in self.search_space.items():
            if ptype == "float":
                if len(args) == 3 and args[2] == "log":
                    params[name] = trial.suggest_float(name, args[0], args[1], log=True)
                else:
                    params[name] = trial.suggest_float(name, *args)
            elif ptype == "int":
                params[name] = trial.suggest_int(name, *args)
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, args[0])
            else:
                raise ValueError(f"Unsupported parameter type: {ptype}")
        return params

    def objective(self, trial):
        params = self.suggest_parameters(trial)
        self.model.set_hp(**params)
        self.model.fit(self.x_train, self.y_train)

        if self.evaluator is not None:
            score = self.evaluator(self.model, trial)
        elif self.x_val is not None and self.y_val is not None and self.metric is not None:
            y_pred = self.model.predict(self.x_val)
            score = self.metric(self.y_val, y_pred)
        else:
            raise ValueError("No evaluator or validation+metric provided.")

        if self.verbose:
            print(f"Trial {trial.number:02}: Params={params} → Score={score:.5f}")

        # 🌟 Speichere bestes Modell sofort, wenn Score besser ist
        if score < self.best_score:
            self.best_score = score
            self.best_model = copy.deepcopy(self.model)
            if self.verbose:
                print(f"[Tuner] New best model saved at trial {trial.number:02} with score {score:.5f}")

        # Optional: Logging
        if self.trial_logger is not None:
            trial_id = f"{self.sampler_type}_{self.run_id:02}_{trial.number:02}"
            best_so_far = self.best_score

            y_pred = trial.user_attrs.get("y_pred")
            y_true = trial.user_attrs.get("y_true")

            if y_pred is None and self.x_val is not None:
                try:
                    y_pred = self.model.predict(self.x_val)
                    y_true = self.y_val
                except Exception as e:
                    print(f"[TrialLogger] Prediction failed: {e}")
                    y_pred, y_true = None, None

            if y_pred is not None and y_true is not None:
                self.trial_logger.log(
                    trial_id=trial_id,
                    method=self.sampler_type,
                    run_id=self.run_id,
                    trial_index=trial.number,
                    rmse=score,
                    best_so_far=best_so_far,
                    y_true=y_true,
                    y_pred=y_pred,
                    params=params
                )

        return score

    def optimize(self):
        self.study = optuna.create_study(direction=self.direction, sampler=self.get_sampler())
        self.study.optimize(self.objective, n_trials=self.n_trials)

        if self.trial_logger is not None:
            from optuna.importance import get_param_importances
            importances = get_param_importances(self.study)
            self.trial_logger.log_importance(importances, self.sampler_type, self.run_id)
            self.trial_logger.log_importance_csv(importances, self.sampler_type, self.run_id)

        return self.study

    def save_best_model(self, path):
        if self.best_model is not None:
            joblib.dump(self.best_model, path)
            print(f"[Tuner] Best model saved to: {path}")
        else:
            print("[Tuner] No best model to save. Did you run optimize()?")

    def report(self):
        print("Best Params:", self.study.best_trial.params)
        print("Best Score:", self.study.best_trial.value)
