from io import BytesIO
import sys

import cv2
from PIL import Image

sys.path.append("src/models/yolact")
import numpy as np
from fastapi import FastAPI, UploadFile, File
import torch
from pred import getModel, predImage

net = getModel()



app = FastAPI()


def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))


@app.post("/detection")
async def read_root(file: UploadFile = File(...)):
    image = load_image_into_numpy_array(await file.read())
    return predImage(net, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


@app.get("/cuda")
async def root():
    try:
        return {
            "cuda-available": torch.cuda.is_available(),
            "cuda-device-count": torch.cuda.device_count(),
            "cuda-device-0": torch.cuda.get_device_name(0)
        }

    except Exception:
        return {"cuda": False}
