from ultralytics import YOLO
import os
import cv2
import shutil
from datetime import datetime
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

fps = 30

def detect(path, selected_classes):
    print("Detection started!")

    # Nasz wyuczony model
    model = YOLO("best.pt")

    class_map = {0: 'stop_sign', 1: 'red_light', 2: 'green_light', 3: 'yellow_light'}
    selected_class_indices = [index for index, name in class_map.items() if name in selected_classes]

    results = model.predict(source=path, conf=0.25, save=False)

    save_dir = "runs/detect/" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    filtered_results = []
    for result in results:
        if hasattr(result, 'boxes'):
            filtered_boxes = [box for box in result.boxes if box.cls in selected_class_indices]
            result.boxes = filtered_boxes
            filtered_results.append(result)

    output_file_path = os.path.join(save_dir, "yolo_output.txt")
    new_objects_path = os.path.join(save_dir, "new_objects.txt")

    with open(output_file_path, 'w') as f_all:
        for result in filtered_results:
            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    f_all.write(f'cls: tensor([{box.cls.item()}])\n')

    line_indices = {}

    with open(output_file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.startswith('cls: tensor(['):
            obj = line.strip()[12:-2]
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
    save_filtered_results_as_video(filtered_results, temp_video_path)

    add_original_audio_to_video(path, temp_video_path, save_video_path)

    beep_path = "beep.wav"
    final_video_with_beep = os.path.join(save_dir, "filtered_output.avi")

    with open(new_objects_path, 'r') as file:
        for line in file:
            beep_start_time = int(line.strip()) / fps
            add_beep_to_video(save_video_path, beep_path, final_video_with_beep, beep_start_time)

    print("Filtered results saved with audio and beep.")

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

    video_clip = VideoFileClip(temp_video_path)

    final_clip = video_clip.set_audio(audio)
    final_clip.write_videofile(final_video_path, codec='libx264', audio_codec='aac')

    os.remove(temp_video_path)

def add_beep_to_video(video_path, beep_path, output_path, beep_start_time):
    video_clip = VideoFileClip(video_path)
    beep_sound = AudioFileClip(beep_path).set_start(beep_start_time)

    if video_clip.audio is not None:
        final_audio = CompositeAudioClip([video_clip.audio, beep_sound])
        final_clip = video_clip.set_audio(final_audio)
    else:
        final_clip = video_clip.set_audio(beep_sound)

    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
