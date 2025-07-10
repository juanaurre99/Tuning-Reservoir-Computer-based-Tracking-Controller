import optuna
import numpy as np
from pyreco.metrics import assign_metric



class Tuner:
    def __init__(
        self,
        model,                  # An already constructed RC model
        search_space,           # Dict: param_name -> (type, *args)
        x_train, y_train,
        x_val, y_val,
        metric='mean_squared_error',
        n_trials=50,
        sampler_type='tpe',     # 'tpe', 'random', 'grid', 'gp'
        num_float_steps=5       # Used only for grid
    ):
        self.model = model
        self.search_space = search_space
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.metric = assign_metric(metric)
        self.n_trials = n_trials
        self.sampler_type = sampler_type
        self.num_float_steps = num_float_steps
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
        elif self.sampler_type == 'gp':
            return optuna.samplers.GPSampler()
        elif self.sampler_type == 'grid':
            grid = self.make_grid()
            return optuna.samplers.GridSampler(grid)
        else:
            raise ValueError(f"Unknown sampler_type: {self.sampler_type}")

    def suggest_parameters(self, trial):
        params = {}
        for name, (ptype, *args) in self.search_space.items():
            if ptype == "float":
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
        self.model.set_hp(**params)   # Use your advanced set_hp!
        self.model.compile(optimizer='ridge', metrics=['mean_squared_error'])
        self.model.fit(self.x_train, self.y_train)
        y_pred = self.model.predict(self.x_val)
        return self.metric(self.y_val, y_pred)

    def optimize(self, direction="minimize"):
        self.study = optuna.create_study(direction=direction, sampler=self.get_sampler())
        self.study.optimize(self.objective, n_trials=self.n_trials)
        return self.study

    def report(self):
        print("Best Params:", self.study.best_trial.params)
        print("Best Score:", self.study.best_trial.value)
