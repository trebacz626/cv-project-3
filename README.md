# cv-project-3

First please run:

pip install -r requirements.txt



## Yolact
### Eval
If you want to run our models please type:

Fo resnet 50 model:

python src/models/yolact/eval.py --trained_model=models/yolact/best_model_adam.pth --config=yolact_resnet50_config --test


For darknet model:

python src/models/yolact/eval.py --trained_model=models/yolact/best_model53.pth --config=yolact_darknet53_config --test

if you want to see the results add --display argument

### Train
If you want to train model on your own please type on of the following:
- dvc repro train_yolact
- dvc repro train_yolact

### API
If you want to ru API build docker image:

docker build . -t yoloapi:1

Than run it:

docker run -p 8000:8000 yoloapi:1

And test with:

python src/api/test_script.py
