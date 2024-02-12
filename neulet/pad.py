import os
import cv2
import math
import numpy as np

from utils import Utils
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

class engine:
    def __init__(self):
        opts = Utils.open_settings_yaml()
        self.FPS = opts['setting']['']
        self.HEIGHT = opts['setting']['height']
        self.WIDTH = opts['setting']['width']
        self.WORKERS = opts['setting']['workers']
        self.SOURCE_PATH = opts['setting']['src_path']
        self.PADDED_SAVE_PATH = opts['setting']['padded_path']
        self.max_duration = 0

    def get_max_duration(self):
        for filename in tqdm(os.listdir(self.SOURCE_PATH), total=len(os.listdir(self.SOURCE_PATH))):
            full_filename = os.path.join(self.SOURCE_PATH, filename)
            cap = cv2.VideoCapture(full_filename)
            duration = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT) // cap.get(cv2.CAP_PROP_FPS))

            if duration >= self.max_duration:
                self.max_duration = duration
                max_file = filename

        print(f"Max duration - {self.max_duration} sec - {max_file}")
        return self.max_duration

    def processing_src(self, filename):
        file_path = os.path.join(self.SOURCE_PATH, filename)
        cap = cv2.VideoCapture(file_path)
        output_path = os.path.join(self.PADDED_SAVE_PATH, filename)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        output = cv2.VideoWriter(output_path, fourcc, self.FPS, (self.HEIGHT, self.WIDTH))

        current_duration =  math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT) // cap.get(cv2.CAP_PROP_FPS))
        target_padding_frame = self.max_duration - current_duration

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (self.HEIGHT, self.WIDTH))
            output.write(frame)

        for _ in range(target_padding_frame):
            output.write(frame)

        cap.release()
        output.release()

    def run_threads(self):
        self.get_max_duration()
        executor = ThreadPoolExecutor(max_workers=self.WORKERS)
        futures = []

        for filename in tqdm(os.listdir(self.SOURCE_PATH), total=len(os.listdir(self.SOURCE_PATH))):
            future = executor.submit(self.processing_src, filename)
            futures.append(future)

        for future in tqdm(futures, total=len(futures)):
            future.result()


if __name__ == '__main__':
    instance = engine()
    instance.run_threads()