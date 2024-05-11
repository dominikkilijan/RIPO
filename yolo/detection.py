from ultralytics import YOLO


def detect(path):
    print("detection!")

    #nasz wyuczony model
    model = YOLO("best.pt")

    #defaultowy model
    #model = YOLO("yolov8m.pt")

    results = model.predict(source=path, conf=0.25, save=True)
    print(results)
