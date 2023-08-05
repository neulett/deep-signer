import os
import cv2
import shutil
import numpy as np
import threading
import mediapipe as mp
from tqdm import tqdm

path = 'C:/Users/ILIXYENILXY/Desktop/Deep-_Signer/dataset/0001~3000(영상)/'
dest_path = './padded_source/'

def get_MaxFrame():
    max_frames = 0
    max_filename = ''
    for file_name in tqdm(os.listdir(path), colour='green'):
        file_path = os.path.join(path, file_name)
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if frames > max_frames:
            max_frames = frames
            max_filename = file_name
    print(f'filename: {max_filename}\nlength: {max_frames}')
    print(f'FPS: {fps}\nHeight: {height}\nWidth {width}')

    return max_frames

def process_video(input_path, output_path, max_frames, file_name):
    file_path = os.path.join(input_path, file_name)
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_count == max_frames:
        output_file_path = os.path.join(output_path, file_name)
        shutil.copyfile(file_path, output_file_path)
    else:
        output_file_path = os.path.join(output_path, file_name)
        output_cap = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                output_cap.write(frame)
            else:
                output_cap.write(frame)
        output_cap.release()

    cap.release()

    del output_cap

def pad_len_threaded(input_path, output_path, max_frames):
    threads = []
    for file_name in tqdm(os.listdir(input_path), colour='blue'):
        t = threading.Thread(target=process_video, args=(input_path, output_path, max_frames, file_name))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

max_frames = get_MaxFrame()
pad_len_threaded(path, dest_path, max_frames)