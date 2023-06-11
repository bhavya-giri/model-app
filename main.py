from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
from fastai.vision.all import *
import skimage
from urllib.request import urlopen
import os

app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=methods,
    allow_headers=headers,
)

learn = load_learner("export.pkl")


@app.get("/")
async def root():
    return {"message": "Welcome to the Garbage Classification API!"}


@app.post("/predict")
async def get_net_image_prediction(image_link: str = ""):
    if image_link == "":
        return {"message": "No image link provided"}
    pred, idx, prob = learn.predict(PILImage.create(urlopen(image_link)))
    return {"predcition": pred, "probability": float(prob[0])}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run(app, host="0.0.0.0", port=port)
