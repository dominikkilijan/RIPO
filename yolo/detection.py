from ultralytics import YOLO
import os
import cv2
import shutil

def detect(path, selected_classes):
    print("detection!")

    # Nasz wyuczony model
    model = YOLO("best.pt")

    # Domyślny model
    # model = YOLO("yolov8m.pt")

    # Mapa indeksów do nazw klas
    class_map = {0: 'stop_sign', 1: 'red_light', 2: 'green_light', 3: 'yellow_light'}
    selected_class_indices = [index for index, name in class_map.items() if name in selected_classes]

    results = model.predict(source=path, conf=0.25, save=False)

    # Stwórz nowy katalog na przefiltrowane wyniki
    save_dir = "runs/detect/filtered_predict"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Filtruj wyniki na podstawie wybranych klas
    filtered_results = []
    for result in results:
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            filtered_boxes = [box for box in boxes if box.cls in selected_class_indices]
            result.boxes = filtered_boxes
            filtered_results.append(result)

    # Zapisz przefiltrowane wyniki jako plik wideo
    save_video_path = os.path.join(save_dir, "filtered_output.avi")
    save_filtered_results_as_video(filtered_results, save_video_path)

    print("Filtered results saved.")

def save_filtered_results_as_video(results, save_path, fps=30):
    height, width = results[0].orig_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for result in results:
        frame = result.plot()
        out.write(frame)

    out.release()
