from gui.MediaPlayerApp import MediaPlayerApp
import torch
import os

from yolo import train, detection


#jakis taki test czy mozna korzystac z gpu
def check_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Używane urządzenie:', device)

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print(root_dir)

    #uczenie modelu
    #train.train_model(root_dir)


    #reczne uruchamianie wykrywania
    #detection.detect(path)

    #odtwarzacz wideo
    app = MediaPlayerApp(root_dir)
    app.mainloop()

    #check_device()
