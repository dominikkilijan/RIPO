import os
import shutil
import tkinter as tk
import vlc
from tkinter import filedialog, messagebox
from datetime import timedelta
#import detection

class MediaPlayerApp(tk.Tk):
    def __init__(self, root_dir):
        super().__init__()
        self.title("Media Player")
        self.geometry("800x600")
        self.configure(bg="#f0f0f0")
        self.initialize_player()
        self.root_dir = root_dir

    def initialize_player(self):
        self.instance = vlc.Instance()
        self.media_player = self.instance.media_player_new()
        self.current_file = None
        self.playing_video = False
        self.video_paused = False
        self.create_widgets()

    def create_widgets(self):
        self.media_canvas = tk.Canvas(self, bg="black", width=800, height=400)
        self.media_canvas.pack(pady=10, fill=tk.BOTH, expand=True)
        self.select_file_button = tk.Button(
            self,
            text="Select File",
            font=("Arial", 12, "bold"),
            command=self.select_file,
        )
        self.select_file_button.pack(pady=5)
        self.time_label = tk.Label(
            self,
            text="00:00:00 / 00:00:00",
            font=("Arial", 12, "bold"),
            fg="#555555",
            bg="#f0f0f0",
        )
        self.time_label.pack(pady=5)
        self.control_buttons_frame = tk.Frame(self, bg="#f0f0f0")
        self.control_buttons_frame.pack(pady=5)
        self.play_button = tk.Button(
            self.control_buttons_frame,
            text="Play",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            command=self.play_video,
        )
        self.play_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.pause_button = tk.Button(
            self.control_buttons_frame,
            text="Pause",
            font=("Arial", 12, "bold"),
            bg="#FF9800",
            fg="white",
            command=self.pause_video,
        )
        self.pause_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.stop_button = tk.Button(
            self.control_buttons_frame,
            text="Stop",
            font=("Arial", 12, "bold"),
            bg="#F44336",
            fg="white",
            command=self.stop,
        )
        self.stop_button.pack(side=tk.LEFT, pady=5)

        self.save_button = tk.Button(
            self.control_buttons_frame,
            text="Save",
            font=("Arial", 12, "bold"),
            bg="#607D8B",
            fg="white",
            command=self.save_video_as,
        )
        self.save_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.progress_bar = VideoProgressBar(
            self, self.set_video_position, bg="#e0e0e0", highlightthickness=0
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

    def save_video_as(self):
        if self.current_file:
            save_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI files", "*.avi")])
            if save_path:
                shutil.copyfile(self.current_file, save_path)
                messagebox.showinfo("Saved", "Video saved successfully!")

    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Media Files", "*.mp4 *.avi *.mkv")]
        )
        print(file_path)
        if file_path:
            selected_classes = self.show_class_selection_dialog()
            if selected_classes:
                from yolo.detection import detect
                detect(file_path, selected_classes)
                print("detection completed!")
                latest_predict_video = self.get_latest_predict_video()
                if latest_predict_video:
                    self.current_file = latest_predict_video
                    self.time_label.config(text="00:00:00 / " + self.get_duration_str())
                    self.play_video()

    def show_class_selection_dialog(self):
        dialog = tk.Toplevel(self)
        dialog.title("Select Classes")
        dialog.geometry("300x300")
        tk.Label(dialog, text="Select classes to detect:").pack(pady=10)

        classes = ["stop_sign", "red_light", "green_light", "yellow_light"]
        self.class_vars = {class_name: tk.BooleanVar(value=True) for class_name in classes}

        for class_name, var in self.class_vars.items():
            tk.Checkbutton(dialog, text=class_name, variable=var).pack(anchor=tk.W)

        def on_submit():
            selected_classes = [class_name for class_name, var in self.class_vars.items() if var.get()]
            dialog.destroy()
            if not selected_classes:
                messagebox.showwarning("No Classes Selected", "You must select at least one class.")
            else:
                self.selected_classes = selected_classes
                dialog.destroy()

        tk.Button(dialog, text="Submit", command=on_submit).pack(pady=20)
        dialog.wait_window()
        return getattr(self, 'selected_classes', None)

    def get_latest_predict_video(self):
        root_folder = os.path.join(self.root_dir, 'runs', 'detect')
        latest_folder = max([os.path.join(root_folder, d) for d in os.listdir(root_folder)], key=os.path.getmtime)
        video_path = os.path.join(latest_folder, "filtered_output.avi")
        if os.path.exists(video_path):
            return video_path
        return None

    def get_duration_str(self):
        if self.playing_video:
            total_duration = self.media_player.get_length()
            total_duration_str = str(timedelta(milliseconds=total_duration))[:-3]
            return total_duration_str
        return "00:00:00"

    def play_video(self):
        if not self.playing_video:
            media = self.instance.media_new(self.current_file)
            self.media_player.set_media(media)
            self.media_player.set_hwnd(self.media_canvas.winfo_id())
            self.media_player.play()
            self.playing_video = True

    def fast_forward(self):
        if self.playing_video:
            current_time = self.media_player.get_time() + 10000
            self.media_player.set_time(current_time)

    def rewind(self):
        if self.playing_video:
            current_time = self.media_player.get_time() - 10000
            self.media_player.set_time(current_time)

    def pause_video(self):
        if self.playing_video:
            if self.video_paused:
                self.media_player.play()
                self.video_paused = False
                self.pause_button.config(text="Pause")
            else:
                self.media_player.pause()
                self.video_paused = True
                self.pause_button.config(text="Resume")

    def stop(self):
        if self.playing_video:
            self.media_player.stop()
            self.playing_video = False
        self.time_label.config(text="00:00:00 / " + self.get_duration_str())

    def set_video_position(self, value):
        if self.playing_video:
            total_duration = self.media_player.get_length()
            position = int((float(value) / 100) * total_duration)
            self.media_player.set_time(position)

    def update_video_progress(self):
        if self.playing_video:
            total_duration = self.media_player.get_length()
            current_time = self.media_player.get_time()
            progress_percentage = (current_time / total_duration) * 100
            self.progress_bar.set(progress_percentage)
            current_time_str = str(timedelta(milliseconds=current_time))[:-3]
            total_duration_str = str(timedelta(milliseconds=total_duration))[:-3]
            self.time_label.config(text=f"{current_time_str} / {total_duration_str}")
        self.after(1000, self.update_video_progress)

class VideoProgressBar(tk.Scale):
    def __init__(self, master, command, **kwargs):
        kwargs["showvalue"] = False
        super().__init__(
            master,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            length=800,
            command=command,
            **kwargs,
        )
        self.bind("<Button-1>", self.on_click)

    def on_click(self, event):
        if self.cget("state") == tk.NORMAL:
            value = (event.x / self.winfo_width()) * 100
            self.set(value)

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    app = MediaPlayerApp(root_dir)
    app.mainloop()