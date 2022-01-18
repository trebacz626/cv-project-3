from io import BytesIO
from tkinter import Image
import sys
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
    return predImage(net,)


@app.get("/cuda")
async def root():
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    return {"cuda": torch.cuda.is_available()}
