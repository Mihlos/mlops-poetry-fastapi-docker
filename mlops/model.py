import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from mlops.schemas.iris import IrisInput


class IrisClassifier:
    def __init__(self):
        self.X, self.y = load_iris(return_X_y=True)
        self.clf = self.train_model()
        self.iris_type = {0: "setosa", 1: "versicolor", 2: "virginica"}

    def train_model(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(n_neighbors=3).fit(self.X, self.y)

    def classify_iris(self, features: dict):
        X = IrisInput(**features)

        prediction = self.clf.predict_proba(
            [[X.sepal_l, X.sepal_w, X.petal_l, X.petal_w]]
        )
        return {
            "class": self.iris_type[np.argmax(prediction)],
            "probability": round(max(prediction[0]), 2),
        }

    def save_model(self):
        joblib.dump(self.clf, "model.pickle")

    def load_model(self):
        return joblib.load("model.pickle")


if __name__ == "__main__":

    """
    Just to save the model and testing.
    """

    model = IrisClassifier()
    model.save_model()
    print(
        model.classify_iris(
            {"sepal_l": 1.1, "sepal_w": 2.1, "petal_l": 3.1, "petal_w": 4.1}
        )
    )

    knn = model.load_model()
    print(knn.predict_proba([[1.1, 2.1, 3.1, 4.1]]))
