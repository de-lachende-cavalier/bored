from pathlib import Path
from datetime import datetime

import joblib
import torch
import json

import numpy as np

from typing import Literal

RUNS_DIR = "runs/"


def get_latest_model(model_type: Literal["clf", "mlp"] = "clf"):
    path = Path(RUNS_DIR)
    dt_format = "%d%m%Y-%H%M%S"

    model_files = [str(file) for file in path.glob(f"{model_type}_*")]
    latest_model_idx = np.argmax(
        [datetime.strptime(f.split("_")[-1], dt_format) for f in model_files]
    )

    if model_type == "clf":
        return joblib.load(model_files[latest_model_idx])
    return torch.load(model_files[latest_model_idx])


def get_latest_vectoriser():
    # this is a separate function because
    # 1) the vectoriser is not dependent on model type
    # 2) we might need a vectorises without a model and vice versa
    path = Path(RUNS_DIR)
    dt_format = "%d%m%Y-%H%M%S"

    vec_files = [str(file) for file in path.glob(f"vec_*")]
    latest_vec = np.argmax(
        [datetime.strptime(f.split("_")[-1], dt_format) for f in vec_files]
    )

    return joblib.load(vec_files[latest_vec])


def get_latest_encmap():
    path = Path(RUNS_DIR)
    dt_format = "%d%m%Y-%H%M%S"

    enc_files = [str(file) for file in path.glob(f"encmap_*")]
    latest_enc = np.argmax(
        [datetime.strptime(f.split("_")[-1], dt_format) for f in enc_files]
    )

    encmap = {}
    with open(enc_files[latest_enc], "r") as ef:
        encmap = json.load(ef)
    return encmap
