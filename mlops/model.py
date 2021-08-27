import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from mlops.schemas.iris import IrisInput


class IrisClassifier:
    def __init__(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.clf = self.train_model()
        self.iris_type = {0: "setosa", 1: "versicolor", 2: "virginica"}

    def train_model(self) -> LogisticRegression:
        return LogisticRegression(
            solver="lbfgs", max_iter=1000, multi_class="multinomial"
        ).fit(self.X, self.y)

    def classify_iris(self, features: dict):
        X = IrisInput(**features)

        prediction = self.clf.predict_proba(
            [[X.sepal_l, X.sepal_w, X.petal_l, X.petal_w]]
        )
        return {
            "class": self.iris_type[np.argmax(prediction)],
            "probability": round(max(prediction[0]), 2),
        }


if __name__ == "__main__":
    pass
