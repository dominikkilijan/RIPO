import ultralytics
from ultralytics import YOLO
import yaml

path = "C:\\Users\\domin\\OneDrive\\Pulpit\\PWR\\RIPO\\RIPO\\data\\validation\\Test.mp4"

def detect():
     # get_classes_from_config(config_path)
    print("detection!")

    # Inicjalizacja modelu YOLO
    # model = YOLO("volov8m.pt")
    model = YOLO("best.pt")

    #print(model.names)

    # indexClass = model.names.index("person")
    # print(indexClass)

    # Analiza obrazu przy użyciu wytrenowanego modelu
    results = model.predict(source=path, conf=0.25, save=True)
    print(results)


# config_path = "C:\\Users\\domin\\OneDrive\\Pulpit\\PWR\\RIPO\\RIPO\\config.yaml"
# def get_classes_from_config(config_path):
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)
#
#     return config['names']
#
# classes = get_classes_from_config(config_path)
# print(classes)