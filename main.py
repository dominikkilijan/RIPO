from gui.MediaPlayerApp import MediaPlayerApp
import torch

from yolo import train, detection


#jakis taki test czy mozna korzystac z gpu
def check_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Używane urządzenie:', device)

path = "C:\\Users\\domin\\OneDrive\\Pulpit\\PWR\\RIPO\\RIPO\\data\\validation\\1s.mp4"

if __name__ == "__main__":
    #uczenie modelu
    #train.trainModel()

    #reczne uruchamianie wykrywania
    #detection.detect(path)

    #odtwarzacz wideo
    app = MediaPlayerApp()
    app.mainloop()

    #check_device()
