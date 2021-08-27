from fastapi import APIRouter
from starlette.responses import JSONResponse

from mlops.model import IrisClassifier

router = APIRouter()


@router.post("/classify_iris")
def classify_iris(iris_features: dict):
    iris_classifier = IrisClassifier()
    return JSONResponse(iris_classifier.classify_iris(iris_features))
