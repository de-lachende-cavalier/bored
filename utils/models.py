from joblib import load
from pathlib import Path
from datetime import datetime

import numpy as np

from typing import Literal


def get_latest_model(mtype: Literal["clf", "mlp"] = "clf"):
    path = Path("models/")
    dt_format = "%d%m%Y-%H%M%S"

    model_files = [str(file) for file in path.glob(f"{mtype}_*")]
    latest = np.argmax(
        [datetime.strptime(f.split("_")[-1], dt_format) for f in model_files]
    )

    return load(model_files[latest])
