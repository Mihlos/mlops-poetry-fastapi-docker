from fastapi import FastAPI
from mlops.routers import routers

app = FastAPI()
app.include_router(routers.router, prefix="/iris")


@app.get("/healthcheck", status_code=200)
async def healthcheck():
    return "Iris classifier is all ready to go!"
