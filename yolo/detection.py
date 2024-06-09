from ultralytics import YOLO
import os
import cv2
import shutil
from datetime import datetime
from moviepy.editor import VideoFileClip, AudioFileClip

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
    save_dir = "runs/detect/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    print(f"Save directory: {save_dir}")  # Debug print
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
    temp_video_path = os.path.join(save_dir, "filtered_output_temp.avi")
    save_video_path = os.path.join(save_dir, "filtered_output_with_audio.avi")
    save_filtered_results_as_video(filtered_results, temp_video_path)

    # Dodaj oryginalny dźwięk do przefiltrowanego wideo
    add_original_audio_to_video(path, temp_video_path, save_video_path)

    print("Filtered results saved with audio.")

def save_filtered_results_as_video(results, save_path, fps=30):
    height, width = results[0].orig_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for result in results:
        frame = result.plot()
        out.write(frame)

    out.release()

def add_original_audio_to_video(original_video_path, temp_video_path, final_video_path):
    # Extract original audio
    original_clip = VideoFileClip(original_video_path)
    audio = original_clip.audio

    # Load the video without audio
    video_clip = VideoFileClip(temp_video_path)

    # Combine video with original audio
    final_clip = video_clip.set_audio(audio)
    final_clip.write_videofile(final_video_path, codec='libx264', audio_codec='aac')

    # Clean up the temporary video file
    os.remove(temp_video_path)
