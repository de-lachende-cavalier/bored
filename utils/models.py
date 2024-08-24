from joblib import load
from pathlib import Path
from datetime import datetime

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

from typing import Literal


def get_latest_model(mtype: Literal["clf", "mlp"] = "clf"):
    path = Path("models/")
    dt_format = "%d%m%Y-%H%M%S"

    model_files = [str(file) for file in path.glob(f"{mtype}_*")]
    latest = np.argmax(
        [datetime.strptime(f.split("_")[-1], dt_format) for f in model_files]
    )

    return load(model_files[latest])


class ProbabilityMixtureClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, large_clf, small_clf, alpha=0.5):
        self.large_clf = large_clf
        self.small_clf = small_clf
        self.weight1 = alpha

    def fit(self, X, y):
        # large_clf is assumed to be fitted
        self.small_clf.fit(X, y)
        return self

    def predict_proba(self, X):
        prob1 = self.large_clf.predict_proba(X)
        prob2 = self.small_clf.predict_proba(X)
        return self.alpha * prob1 + (1 - self.alpha) * prob2

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
