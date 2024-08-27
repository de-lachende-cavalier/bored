from joblib import load
from pathlib import Path
from datetime import datetime

import numpy as np

from typing import Literal


def get_latest_model(model_type: Literal["clf", "mlp"] = "clf"):
    path = Path("saved_models/")
    dt_format = "%d%m%Y-%H%M%S"

    model_files = [str(file) for file in path.glob(f"{model_type}_*")]
    latest_model = np.argmax(
        [datetime.strptime(f.split("_")[-1], dt_format) for f in model_files]
    )

    return load(model_files[latest_model])


def get_latest_vectoriser():
    # this is a separate function because
    # 1) the vectoriser is not dependent on model type
    # 2) we might need a vectorises without a model and vice versa
    path = Path("saved_models/")
    dt_format = "%d%m%Y-%H%M%S"

    vec_files = [str(file) for file in path.glob(f"vec_*")]
    latest_vec = np.argmax(
        [datetime.strptime(f.split("_")[-1], dt_format) for f in vec_files]
    )

    return load(vec_files[latest_vec])
