# init_and_save_models.py

import os
import joblib
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer

def make_model(input_shape=(1, 8), output_shape=(1, 2), seed=42):
    model = RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(RandomReservoirLayer(
        nodes=200,
        activation='tanh',
        fraction_input=0.5,
        seed=seed,
    ))
    model.add(ReadoutLayer(output_shape))
    model.compile(optimizer='ridge', metrics=['mse'])
    return model

def init_and_save_models(n_models=30, out_dir="saved_reservoirs"):
    os.makedirs(out_dir, exist_ok=True)
    input_shape = (1, 8)
    output_shape = (1, 2)

    for i in range(n_models):
        model = make_model(input_shape, output_shape, seed=1000 + i)
        model_path = os.path.join(out_dir, f"reservoir_{i}.joblib")
        joblib.dump(model, model_path)
        print(f"[Init] Saved model {i} → {model_path}")

if __name__ == "__main__":
    init_and_save_models()
