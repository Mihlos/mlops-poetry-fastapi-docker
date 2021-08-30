import numpy as np
import pickle
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
        with open("model.pkl", "wb") as model:
            pickle.dump(self.clf, model)


if __name__ == "__main__":
    model = IrisClassifier()
    model.save_model()
