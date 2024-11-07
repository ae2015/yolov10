import os, sys
os.chdir("/content/gdrive/MyDrive/code/yolov10") # os.chdir("/Users/alex/Documents/Code/yolov10")
sys.path.append('.')
                
from ultralytics import settings

# Update the datasets directory setting
settings.update({"datasets_dir": "/content/gdrive/MyDrive/datasets"}) # settings.update({"datasets_dir": "."})

from ultralytics import YOLO  #  Reads settings

# Load YOLOv10n model from scratch
model = YOLO("yolov10n.yaml")

# Train the model
model.train(data="/content/gdrive/MyDrive/datasets/try2/doclaynet.yaml", epochs=100, imgsz=640)
# model.train(data="temp/data/try1/doclaynet.yaml", epochs=10, imgsz=640)
