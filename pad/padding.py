import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

path = 'C:/Users/ILIXYENILXY/Desktop/Deep-_Signer/dataset/0001~3000(영상)/'
dest_path = './padded_source/'

def get_MaxFrame():
    max_frames = 0
    for file_name in tqdm(os.listdir(path), colour='green'):
        file_path = os.path.join(path, file_name)
        cap = cv2.VideoCapture(file_path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frames > max_frames:
            max_frames = frames
    return max_frames
max_frame = get_MaxFrame()

def pad_length():
    for file_name in tqdm(os.listdir(path, colour='blue')):
        file_path = os.path.join(path, file_name)
        cap = cv2.VideoCapture(file_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    # 해당 영상의 프레임 구하기.
        fps = int(cap.get(cv2.CAP_PROP_FPS))    # 해당 영상의 FPS 구하기.
        