import os
import cv2
import math
import argparse
import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

class engine:
    def __init__(self):
        self.default_source_path = "../local/source"
        self.default_pad_source_path = '../local/pad_source'
        self.file_list = []
        self.default_H = 800
        self.default_W = 600
        self.max_duration = 0

    def get_max_duration(self):
        max_file = ''
        self.file_list = os.listdir(SOURCE)

        for filename in self.file_list:
            file_path = os.path.join(SOURCE, filename)
            cap = cv2.VideoCapture(file_path)

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            duration = math.ceil(frame_count // frame_rate)  # duration = frame_count // frame_rate

            if duration >= self.max_duration:
                self.max_duration = duration
                max_file = filename

        return self.max_duration, max_file

    def process_video(self, filename):
        file_path = os.path.join(SOURCE, filename)
        cap = cv2.VideoCapture(file_path)
        output_path = os.path.join(DEST, filename)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, fourcc, FPS, (HEIGHT, WIDTH))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (HEIGHT, WIDTH))
            output.write(frame)

    def padding_src(self):
        print(f"Pad workthreads start.")
        if HEIGHT == 0 or WIDTH == 0:
            raise ValueError(f"Source resize ValueError {HEIGHT},{WIDTH}")

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            list(tqdm(executor.map(self.process_video, self.file_list), total=len(self.file_list), desc="Padding Progress"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', default='../video/source/', type=str)
    parser.add_argument('-dest', '--dest', default='../video/padded_source/', type=str)
    parser.add_argument('-height', '--height', default=800, type=int)
    parser.add_argument('-width', '--width', default=600, type=int)
    parser.add_argument('-fps', '--fps', default=30.0, type=float)
    parser.add_argument('-workers', '--workers', type=int)
    args = parser.parse_args()

    SOURCE = args.source
    DEST = args.dest
    HEIGHT = args.height
    WIDTH = args.width
    FPS = args.fps
    WORKERS = args.workers

    instance = engine()
    max_duration, max_file = instance.get_max_duration()
    print(f"Max duration - {max_duration} sec - {max_file}")
    instance.padding_src()
