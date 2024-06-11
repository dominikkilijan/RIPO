# import logging
# from ultralytics import YOLO
# import os
# import cv2
# import shutil
# from datetime import datetime
# from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
# import re
# from subprocess import run
# import subprocess
# import io
# import sys
#
# fps = 30
#
# # Konfiguracja logowania
# log_file_path = "output.log"
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
#
# def detect(path, selected_classes):
#     captured_output = io.StringIO()
#     sys.stdout = captured_output
#     logging.info("detection!")
#
#     # Nasz wyuczony model
#     model = YOLO("best.pt")
#
#     # Domyślny model
#     # model = YOLO("yolov8m.pt")
#
#     # Mapa indeksów do nazw klas
#     class_map = {0: 'stop_sign', 1: 'red_light', 2: 'green_light', 3: 'yellow_light'}
#     selected_class_indices = [index for index, name in class_map.items() if name in selected_classes]
#
#     # Uruchomienie detekcji za pomocą subprocess i przechwytywanie wyjścia
#     command = ["python", "-m", "ultralytics", "detect", "--source", path, "--weights", "best.pt", "--conf", "0.25"]
#     process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#
#     save_dir = "runs/detect/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
#     logging.info(f"Save directory: {save_dir}")
#     if os.path.exists(save_dir):
#         shutil.rmtree(save_dir)
#     os.makedirs(save_dir)
#
#
#
#     # Filtruj wyniki na podstawie wybranych klas
#     results = model.predict(source=path, conf=0.25, save=False)
#     filtered_results = []
#     for result in results:
#         if hasattr(result, 'boxes'):
#             boxes = result.boxes
#             print("boxes:")
#             print(boxes)
#             filtered_boxes = [box for box in boxes if box.cls in selected_class_indices]
#             result.boxes = filtered_boxes
#             filtered_results.append(result)
#
#     output_file_path = os.path.join(save_dir, "yolo_output.txt")
#     new_objects_path = os.path.join(save_dir, "new_objects.txt")
#
#     output = captured_output.getvalue()
#     filtered_lines = [line for line in output.split('\n') if line.startswith('cls: tensor([')]
#
#     with open(output_file_path, 'w') as f_all:
#         for line in filtered_lines:
#             f_all.write(line + '\n')
#
#     # Za moment mozna sie pozbyc zapisywania do plikow
#     with open(output_file_path, 'r') as file:
#         lines = file.readlines()
#
#     # Inicjalizuj zmienne
#     object_counts = {}  # Słownik do przechowywania liczników wystąpień obiektów
#     line_indices = {}  # Słownik do przechowywania numerów linii dla obiektów
#
#     # Iteruj przez linie
#     for i, line in enumerate(lines):
#         # Sprawdź czy linia zawiera tensor
#         if line.startswith('cls: tensor(['):
#             # Wyciągnij obiekty z linii
#             objects = line.strip()[12:-2].split(', ')
#
#             # Iteruj przez obiekty w linii
#             for obj in objects:
#                 if obj:  # Sprawdź, czy obiekt nie jest pusty
#                     # Jeśli obiekt występuje po raz pierwszy, zapisz numer linii
#                     if obj not in line_indices:
#                         line_indices[obj] = [i]  # Zapisz pierwsze wystąpienie obiektu
#                     elif i - line_indices[obj][
#                         -1] == 1:  # Sprawdź, czy obiekt wystąpił w kolejnej linii po poprzednim wystąpieniu
#                         line_indices[obj].append(i)  # Zapisz numer linii dla obiektu
#                     else:
#                         line_indices[obj] = [
#                             i]  # Jeśli przerwa między wystąpieniami jest większa niż 1, zaczynamy od nowa
#
#     # Sprawdź, które obiekty wystąpiły pięć razy z rzędu
#     valid_objects = {obj: indices for obj, indices in line_indices.items() if
#                      len(indices) >= 5 and all(indices[j] == indices[j - 1] + 1 for j in range(1, 5))}
#
#     # Zapisz wyniki do pliku
#     with open(new_objects_path, 'w') as file:
#         for obj, indices in valid_objects.items():
#             for index in indices[:1]:  # Zapisz tylko pierwsze pięć wystąpień
#                 file.write(f'{index}\n')
#
#     print("Monitorowane obiekty zostały zapisane do pliku 'new_objects.txt'.")
#
#     logging.info("YOLO output lines saved to: %s", output_file_path)
#     process.wait()
#
#     sys.stdout = sys.__stdout__
#
#     # Zapisz przefiltrowane wyniki jako plik wideo
#     temp_video_path = os.path.join(save_dir, "filtered_output_temp.avi")
#     save_video_path = os.path.join(save_dir, "filtered_output_with_audio.avi")
#     save_filtered_results_as_video(filtered_results, temp_video_path)
#
#     # Dodaj oryginalny dźwięk do przefiltrowanego wideo
#     add_original_audio_to_video(path, temp_video_path, save_video_path)
#
#     # Dodaj sygnał dźwiękowy beep
#     beep_path = "beep.wav"
#     final_video_with_beep = os.path.join(save_dir, "filtered_output_with_audio_and_beep.avi")
#
#
#     add_beep_to_video(save_video_path, beep_path, final_video_with_beep)
#
#     logging.info("Filtered results saved with audio and beep.")
#
#
# def save_filtered_results_as_video(results, save_path, fps=fps):
#     height, width = results[0].orig_img.shape[:2]
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
#
#     for result in results:
#         frame = result.plot()
#         out.write(frame)
#
#     out.release()
#
# def add_original_audio_to_video(original_video_path, temp_video_path, final_video_path):
#     # Extract original audio
#     original_clip = VideoFileClip(original_video_path)
#     audio = original_clip.audio
#
#     # Load the video without audio
#     video_clip = VideoFileClip(temp_video_path)
#
#     # Combine video with original audio
#     final_clip = video_clip.set_audio(audio)
#     final_clip.write_videofile(final_video_path, codec='libx264', audio_codec='aac')
#
#     # Clean up the temporary video file
#     os.remove(temp_video_path)
#
# def add_beep_to_video(video_path, beep_path, output_path, beep_start_time):
#     # Load the video and beep audio files
#     video_clip = VideoFileClip(video_path)
#     beep_sound = AudioFileClip(beep_path).set_start(beep_start_time)
#
#     # Combine the original audio with the beep sound
#     if video_clip.audio is not None:
#         final_audio = CompositeAudioClip([video_clip.audio, beep_sound])
#         final_clip = video_clip.set_audio(final_audio)
#     else:
#         final_clip = video_clip.set_audio(beep_sound)
#
#     # Write the final video to a file
#     final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
#

import logging
from ultralytics import YOLO
import os
import cv2
import shutil
from datetime import datetime
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
import re
from subprocess import run
import subprocess
import io
import sys

fps = 30

# Konfiguracja logowania
log_file_path = "output.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])

def detect(path, selected_classes):
    captured_output = io.StringIO()
    sys.stdout = captured_output
    logging.info("detection!")

    # Nasz wyuczony model
    model = YOLO("best.pt")

    # Domyślny model
    # model = YOLO("yolov8m.pt")

    # Mapa indeksów do nazw klas
    class_map = {0: 'stop_sign', 1: 'red_light', 2: 'green_light', 3: 'yellow_light'}
    selected_class_indices = [index for index, name in class_map.items() if name in selected_classes]

    # Uruchomienie detekcji za pomocą subprocess i przechwytywanie wyjścia
    command = ["python", "-m", "ultralytics", "detect", "--source", path, "--weights", "best.pt", "--conf", "0.25"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    save_dir = "runs/detect/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    logging.info(f"Save directory: {save_dir}")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Filtruj wyniki na podstawie wybranych klas
    results = model.predict(source=path, conf=0.25, save=False)
    filtered_results = []
    for result in results:
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            print("boxes:")
            print(boxes)
            filtered_boxes = [box for box in boxes if box.cls in selected_class_indices]
            result.boxes = filtered_boxes
            filtered_results.append(result)

    output_file_path = os.path.join(save_dir, "yolo_output.txt")
    new_objects_path = os.path.join(save_dir, "new_objects.txt")

    output = captured_output.getvalue()
    filtered_lines = [line for line in output.split('\n') if line.startswith('cls: tensor([')]

    with open(output_file_path, 'w') as f_all:
        for line in filtered_lines:
            f_all.write(line + '\n')

    # Za moment mozna sie pozbyc zapisywania do plikow
    with open(output_file_path, 'r') as file:
        lines = file.readlines()

    # Inicjalizuj zmienne
    object_counts = {}  # Słownik do przechowywania liczników wystąpień obiektów
    line_indices = {}  # Słownik do przechowywania numerów linii dla obiektów

    # Iteruj przez linie
    for i, line in enumerate(lines):
        # Sprawdź czy linia zawiera tensor
        if line.startswith('cls: tensor(['):
            # Wyciągnij obiekty z linii
            objects = line.strip()[12:-2].split(', ')

            # Iteruj przez obiekty w linii
            for obj in objects:
                if obj:  # Sprawdź, czy obiekt nie jest pusty
                    # Jeśli obiekt występuje po raz pierwszy, zapisz numer linii
                    if obj not in line_indices:
                        line_indices[obj] = [i]  # Zapisz pierwsze wystąpienie obiektu
                    elif i - line_indices[obj][
                        -1] == 1:  # Sprawdź, czy obiekt wystąpił w kolejnej linii po poprzednim wystąpieniu
                        line_indices[obj].append(i)  # Zapisz numer linii dla obiektu
                    else:
                        line_indices[obj] = [
                            i]  # Jeśli przerwa między wystąpieniami jest większa niż 1, zaczynamy od nowa

    # Sprawdź, które obiekty wystąpiły pięć razy z rzędu
    valid_objects = {obj: indices for obj, indices in line_indices.items() if
                     len(indices) >= 5 and all(indices[j] == indices[j - 1] + 1 for j in range(1, 5))}

    # Zapisz wyniki do pliku
    with open(new_objects_path, 'w') as file:
        for obj, indices in valid_objects.items():
            for index in indices[:1]:  # Zapisz tylko pierwsze pięć wystąpień
                file.write(f'{index}\n')

    print("Monitorowane obiekty zostały zapisane do pliku 'new_objects.txt'.")

    logging.info("YOLO output lines saved to: %s", output_file_path)
    process.wait()

    sys.stdout = sys.__stdout__

    # Zapisz przefiltrowane wyniki jako plik wideo
    temp_video_path = os.path.join(save_dir, "filtered_output_temp.avi")
    save_video_path = os.path.join(save_dir, "filtered_output_with_audio.avi")
    save_filtered_results_as_video(filtered_results, temp_video_path)

    # Dodaj oryginalny dźwięk do przefiltrowanego wideo
    add_original_audio_to_video(path, temp_video_path, save_video_path)

    # Dodaj sygnał dźwiękowy beep
    beep_path = "beep.wav"
    final_video_with_beep = os.path.join(save_dir, "filtered_output_with_audio_and_beep.avi")

    # Iteruj przez każdą linię w pliku new_objects_path
    with open(new_objects_path, 'r') as file:
        for line in file:
            beep_start_time = int(line.strip()) / fps
            add_beep_to_video(save_video_path, beep_path, final_video_with_beep, beep_start_time)

    logging.info("Filtered results saved with audio and beep.")

def save_filtered_results_as_video(results, save_path, fps=fps):
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

def add_beep_to_video(video_path, beep_path, output_path, beep_start_time):
    # Load the video and beep audio files
    video_clip = VideoFileClip(video_path)
    beep_sound = AudioFileClip(beep_path).set_start(beep_start_time)

    # Combine the original audio with the beep sound
    if video_clip.audio is not None:
        final_audio = CompositeAudioClip([video_clip.audio, beep_sound])
        final_clip = video_clip.set_audio(final_audio)
    else:
        final_clip = video_clip.set_audio(beep_sound)

    # Write the final video to a file
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
