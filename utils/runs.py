from pathlib import Path
from datetime import datetime

import joblib
import torch
import json

import numpy as np

from typing import Literal

RUNS_DIR = "runs/"
DT_FORMAT = "%d%m%Y-%H%M%S"


def get_latest_model(model_type: Literal["clf", "mlp"] = "clf"):
    path = Path(RUNS_DIR)

    model_files = [str(file) for file in path.glob(f"{model_type}_*")]
    latest_model_idx = np.argmax(
        # the first 15 characters contain the datatime string
        # (this is not the most robust approach, but it keeps things simple)
        [datetime.strptime(f.split("_")[-1][:15], DT_FORMAT) for f in model_files]
    )

    if model_type == "clf":
        return joblib.load(model_files[latest_model_idx])
    return torch.load(model_files[latest_model_idx], weights_only=True)


def get_latest_vectoriser():
    # this is a separate function because
    # 1) the vectoriser is not dependent on model type
    # 2) we might need a vectorises without a model and vice versa
    path = Path(RUNS_DIR)

    vec_files = [str(file) for file in path.glob(f"vec_*")]
    latest_vec = np.argmax(
        [datetime.strptime(f.split("_")[-1][:15], DT_FORMAT) for f in vec_files]
    )

    return joblib.load(vec_files[latest_vec])


def get_latest_encmap():
    path = Path(RUNS_DIR)

    enc_files = [str(file) for file in path.glob(f"encmap_*")]
    latest_enc = np.argmax(
        [datetime.strptime(f.split("_")[-1][:15], DT_FORMAT) for f in enc_files]
    )

    encmap = {}
    with open(enc_files[latest_enc], "r") as ef:
        encmap = json.load(ef)
    return encmap
