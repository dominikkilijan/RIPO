import ultralytics
from ultralytics import YOLO
import yaml

path = "C:\\Users\\domin\\OneDrive\\Pulpit\\PWR\\RIPO\\RIPO\\data\\validation"

def detect():
     # get_classes_from_config(config_path)
    print("detection!")

    model = YOLO("best.pt")

    # Analiza obrazu przy użyciu wytrenowanego modelu
    results = model.predict(source=path, conf=0.25, save=True)
    print(results)