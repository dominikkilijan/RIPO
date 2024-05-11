from ultralytics import YOLO


def train_model():
    print("yolo!")

    model = YOLO("yolov8m.pt")
    model.train(data="config.yaml", epochs=1)

    print("yolo ended")
