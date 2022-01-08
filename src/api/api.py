from fastapi import FastAPI
import torch


app = FastAPI()


@app.get("/")
async def root():
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    return {"message": "Hello World"}
