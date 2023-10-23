import os
import cv2
import math
import argparse
import numpy as np

class pad_tool:
    def __init__(self):
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.MODE = MODE
        self.max_duration = 0

    def get_max_duration(self, path):
        max_file = []
        self.file_list = os.listdir(path)

        for filename in self.file_list:
            file_path = os.path.join(path, filename)
            cap = cv2.VideoCapture(file_path)

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            duration = math.ceil(frame_count // frame_rate)

            if duration >= self.max_duration:
                self.max_duration = duration
                max_file = filename

        print(f"Get Max Duration - {self.max_duration} sec.\nFilename - {max_file}")
        print(f"pad workthreads start.")

        return self.max_duration, max_file

    def pad_length(self):
        for filename in self.file_list:
            file_path = os.path.join(SOURCE, filename)
            cap = cv2.VideoCapture(file_path)
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            duration = frame_count / frame_rate
            target_duration = self.max_duration

        # np.pad(cap, (), 'constant', constant_values=0)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', required=True, type=str)
    parser.add_argument('-dest', '--dest', required=True, type=str)
    parser.add_argument('-h', '--height', default=800, type=int)
    parser.add_argument('-w', '--width', default=600, type=int)
    parser.add_argument('-mode', '--mode', default=pad_mode, type=str)

    args = parser.parse_args()

    SOURCE = args.source
    DEST = args.dest
    HEIGHT = args.height
    WIDTH = args.width
    MODE = args.mode

    print(args.height)
    print(args.width)
    print(args.mode)

