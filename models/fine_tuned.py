import numpy as np

from bornrule import BornClassifier
from sklearn.utils.multiclass import unique_labels


class FineTunedBornClassifier(BornClassifier):
    def __init__(
        self,
        pretrained_model: BornClassifier,
        *,
        n_classes: int,
        a: float = 0.5,
        b: float = 1.0,
        h: float = 1.0,
        learning_rate: float = 0.1,
    ):
        super().__init__(a=a, b=b, h=h)
        self.pretrained_model = pretrained_model
        self.n_classes = n_classes
        self.learning_rate = learning_rate

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        X, y = self._sanitize(X, y)

        # due to little training data we might get inconsistent shapes if we didn't do the below
        if len(self._unique_labels(y)) != self.n_classes:
            classes = unique_labels(np.arange(0, self.n_classes))

        first_call = self._check_partial_fit_first_call(classes)
        if first_call:
            self.corpus_ = self.pretrained_model.corpus_
            self.n_features_in_ = self.pretrained_model.n_features_in_

        if not self._check_encoded(y):
            y = self._one_hot_encoding(y)

        if sample_weight is not None:
            sample_weight = self._check_sample_weight(sample_weight, X)
            y = self._multiply(y, sample_weight.reshape(-1, 1))

        update = X.T @ self._multiply(y, self._power(self._sum(X, axis=1), -1))
        self.corpus_ += self.learning_rate * (update - self.corpus_)

        return self
