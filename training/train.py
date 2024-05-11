# import ultralytics
from ultralytics import YOLO

def trainModel():
    print("training!")
    # Load a model
    model = YOLO("yolov8m.pt")

    # Use the model
    results = model.train(data="config.yaml", epochs=1)  # train the model

    print("training ended")