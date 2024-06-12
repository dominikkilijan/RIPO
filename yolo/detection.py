import logging
from ultralytics import YOLO
import os
import cv2
import shutil
from datetime import datetime
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

fps = 30

default_colors = {
    'stop_sign': (0, 0, 255),  # Blue
    'red_light': (255, 0, 0),  # Red
    'green_light': (0, 255, 0),  # Green
    'yellow_light': (0, 255, 255)  # Yellow
}

high_contrast_colors = {
    'stop_sign': (0, 0, 0),  # Black
    'red_light': (255, 255, 255),  # White
    'green_light': (255, 255, 0),  # Cyan
    'yellow_light': (0, 0, 255)  # Blue
}

colorblind_friendly_colors = {
    'stop_sign': (0, 0, 255),  # Blue
    'red_light': (255, 165, 0),  # Orange
    'green_light': (0, 128, 0),  # Dark Green
    'yellow_light': (75, 0, 130)  # Indigo
}

class_map = {0: 'stop_sign', 1: 'red_light', 2: 'green_light', 3: 'yellow_light'}

def detect(path, selected_classes, color_scheme='default', add_beep=True):
    print("Detection started!")

    # Wybierz zestaw kolorów na podstawie wybranego schematu
    if color_scheme == 'high_contrast':
        colors = high_contrast_colors
    elif color_scheme == 'colorblind_friendly':
        colors = colorblind_friendly_colors
    else:
        colors = default_colors

    model = YOLO("best.pt")

    # Mapa indeksów do nazw klas
    selected_class_indices = [index for index, name in class_map.items() if name in selected_classes]

    # Uruchomienie detekcji
    results = model.predict(source=path, conf=0.55, save=False)

    save_dir = "runs/detect/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # Filtruj wyniki na podstawie wybranych klas
    filtered_results = []
    yolo_output_lines = []
    line_indices = {}

    filtered_video_results = []
    for result in results:
        if hasattr(result, 'boxes'):
            filtered_video_boxes = [box for box in result.boxes if box.cls in selected_class_indices]
            result.boxes = filtered_video_boxes  # Zmienione z 'result.video_boxes' na 'result.boxes'
            filtered_video_results.append(result)

    for i, result in enumerate(results):
        if hasattr(result, 'boxes'):
            filtered_boxes = [box for box in result.boxes if box.cls in selected_class_indices]
            if filtered_boxes:
                result.boxes = filtered_boxes
                filtered_results.append(result)
                objects = ', '.join([str(box.cls.item()) for box in filtered_boxes])
                line = f'cls: tensor([{objects}])'
                yolo_output_lines.append(line)
                for box in filtered_boxes:
                    obj = str(box.cls.item())
                    if obj not in line_indices:
                        line_indices[obj] = [i]
                    elif i - line_indices[obj][-1] == 1:
                        line_indices[obj].append(i)
                    else:
                        line_indices[obj] = [i]

    # Sprawdź, które obiekty wystąpiły pięć razy z rzędu
    valid_objects = {obj: indices for obj, indices in line_indices.items() if len(indices) >= 5 and all(indices[j] == indices[j - 1] + 1 for j in range(1, 5))}

    new_objects_lines = [f"{obj} {indices[0]}" for obj, indices in valid_objects.items()]

    # Zapisz wyniki do plików
    output_file_path = os.path.join(save_dir, "yolo_output.txt")
    new_objects_path = os.path.join(save_dir, "new_objects.txt")

    with open(output_file_path, 'w') as f_all:
        f_all.write('\n'.join(yolo_output_lines))

    with open(new_objects_path, 'w') as file:
        file.write('\n'.join(new_objects_lines))

    with open(output_file_path, 'r') as file:
        lines = file.readlines()

    line_indices = {}

    for i, line in enumerate(lines):
        if line.startswith('cls: tensor(['):
            objects = line.strip()[12:-2].split(', ')
            for obj in objects:
                if obj:
                    if obj not in line_indices:
                        line_indices[obj] = [i]
                    elif i - line_indices[obj][-1] == 1:
                        line_indices[obj].append(i)
                    else:
                        line_indices[obj] = [i]

    valid_objects = {obj: indices for obj, indices in line_indices.items() if len(indices) >= 5 and all(indices[j] == indices[j - 1] + 1 for j in range(1, 5))}

    with open(new_objects_path, 'w') as file:
        for obj, indices in valid_objects.items():
            for index in indices[:1]:
                file.write(f'{index}\n')

    temp_video_path = os.path.join(save_dir, "filtered_output_temp.avi")
    save_video_path = os.path.join(save_dir, "filtered_output.avi")
    save_filtered_results_as_video(filtered_video_results, temp_video_path, colors)

    add_original_audio_to_video(path, temp_video_path, save_video_path)

    if add_beep:
        final_video_with_beep = os.path.join(save_dir, "filtered_output_with_beep.avi")
        beep_intervals = detect_intervals(yolo_output_lines, fps)
        add_beep_to_video(save_video_path, "beep-02.wav", final_video_with_beep, beep_intervals)
    else:
        final_video_with_beep = save_video_path

def save_filtered_results_as_video(results, save_path, colors, fps=fps):
    height, width = results[0].orig_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    box_thickness = 3
    font_scale = 1.5
    for result in results:
        frame = result.orig_img.copy()
        for box in result.boxes:
            cls_index = box.cls.item()
            cls_name = class_map[int(cls_index)]
            color = colors[cls_name]
            confidence = box.conf.item() * 100
            label = f'{cls_name} {confidence:.2f}%'
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, box_thickness)
        out.write(frame)
    out.release()

def add_original_audio_to_video(original_video_path, temp_video_path, final_video_path):
    original_clip = VideoFileClip(original_video_path)
    audio = original_clip.audio

    video_clip = VideoFileClip(temp_video_path)

    final_clip = video_clip.set_audio(audio)
    final_clip.write_videofile(final_video_path, codec='libx264', audio_codec='aac')

    os.remove(temp_video_path)

def detect_intervals(yolo_output_lines, fps):
    object_intervals = {}
    current_objects = set()

    for frame_idx, line in enumerate(yolo_output_lines):
        detected_objects = set(line.strip()[12:-2].split(', '))
        new_objects = detected_objects - current_objects
        missing_objects = current_objects - detected_objects

        for obj in new_objects:
            if obj not in object_intervals:
                object_intervals[obj] = []
            object_intervals[obj].append([frame_idx / fps])

        for obj in missing_objects:
            if obj in object_intervals and len(object_intervals[obj][-1]) == 1:
                object_intervals[obj][-1].append(frame_idx / fps)

        current_objects = detected_objects

    for obj in current_objects:
        if obj in object_intervals and len(object_intervals[obj][-1]) == 1:
            object_intervals[obj][-1].append(len(yolo_output_lines) / fps)
            beep_intervals = []
    for intervals in object_intervals.values():
        for start, end in intervals:
            beep_intervals.append((start, end))

    return beep_intervals

def add_beep_to_video(video_path, beep_path, output_path, beep_intervals):
    video_clip = VideoFileClip(video_path)
    try:
        beep_sound = AudioFileClip(beep_path)
        beep_duration = beep_sound.duration
        if beep_duration == 0:
            raise ValueError("Beep sound duration is zero.")
    except Exception as e:
        print(f"Error loading beep sound: {e}")
        return
    
    beep_clips = []
    for start_time, end_time in beep_intervals:
        interval_duration = end_time - start_time
        if interval_duration > beep_duration:
            interval_duration = beep_duration
        print(f"Adding beep from {start_time} to {start_time + interval_duration}, duration: {interval_duration}")
        beep_clip = beep_sound.subclip(0, interval_duration).set_start(start_time)
        beep_clips.append(beep_clip)

    if video_clip.audio is not None:
        final_audio = CompositeAudioClip([video_clip.audio] + beep_clips)
        final_clip = video_clip.set_audio(final_audio)
    else:
        final_clip = video_clip.set_audio(CompositeAudioClip(beep_clips))

    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
