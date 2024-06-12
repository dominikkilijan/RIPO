import os
from ultralytics import YOLO

def train_model(root_dir):
    config_path = os.path.join(root_dir, 'config.yaml')

    print("yolo!")

    model = YOLO("best.pt")
    model.train(data=config_path, epochs=1)

    print("yolo ended")
