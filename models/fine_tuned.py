from bornrule import BornClassifier


class FineTunedBornClassifier(BornClassifier):
    def __init__(
        self,
        pretrained_model: BornClassifier,
        a: float = 0.5,
        b: float = 1.0,
        h: float = 1.0,
        learning_rate: float = 0.1,
    ):
        super().__init__(a=a, b=b, h=h)
        self.pretrained_model = pretrained_model
        self.learning_rate = learning_rate

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        X, y = self._sanitize(X, y)

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
