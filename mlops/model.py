import numpy as np
import pickle

from mlops.schemas.iris import IrisInput


class IrisClassifier:
    def __init__(self):
        self.clf = self.load_model()
        self.iris_type = {0: "setosa", 1: "versicolor", 2: "virginica"}

    def load_model(self):
        with open("model.pkl") as model:
            pickle.load(model)

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
    model = IrisClassifier()
    print(model.clf)
