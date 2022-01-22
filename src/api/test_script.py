import requests


f = open("data/TrashCan/val/vid_000002_frame0000013.jpg", 'rb')
files = {"file": (f.name, f, "multipart/form-data")}
res = requests.post(url="http://127.0.0.1:8000/detection", files=files).json()
print(res)
print(res.keys())
print(res["score"])

