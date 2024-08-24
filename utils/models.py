from joblib import load
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from sklearn.base import BaseEstimator, ClassifierMixin

from typing import Literal


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


def get_latest_model(mtype: Literal["clf", "mlp"] = "clf"):
    path = Path("models/")
    dt_format = "%d%m%Y-%H%M%S"

    model_files = [str(file) for file in path.glob(f"{mtype}_*")]
    latest = np.argmax(
        [datetime.strptime(f.split("_")[-1], dt_format) for f in model_files]
    )

    return load(model_files[latest])


def get_mlp_data(X, y, born_clf, features, mappings):
    columns = mappings["ner_tag"]

    mlp_X = []
    mlp_y = []
    for i, x in enumerate(X):
        x_explanation = pd.DataFrame(
            born_clf.explain(x).toarray(), index=features, columns=columns
        )
        most_likely_y = x_explanation.sum().idxmax()

        num_ner_tag = list(mappings["ner_tag"]).index(most_likely_y)
        mlp_features = x_explanation[most_likely_y].to_list()
        mlp_features.append(num_ner_tag)

        mlp_X.append(mlp_features)
        mlp_y.append(y[i])
        return torch.tensor(mlp_X), torch.tensor(mlp_y)
