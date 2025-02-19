# https://docs.ultralytics.com/models/yolo11/

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.predict(source = "0",show=True,conf=0.6)
