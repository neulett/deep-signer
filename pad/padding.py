import os
import cv2
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

path = input("Source path: ")
dest_path = input("Dest path: ")
target_width = int(input("target width: "))
target_height = int(input("target height: "))

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
    print(f'FPS: {fps}\nHeight: {height}\nWidth: {width}')

    return max_frames

def process_video(input_path, output_path, max_frames, file_name, target_width, target_height):
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
        output_cap = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_width, target_height))

        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (target_width, target_height))
            output_cap.write(resized_frame)

        output_cap.release()

    cap.release()
    del output_cap

def working_threads_pool(input_path, output_path, max_frames, target_width, target_height):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file_name in tqdm(os.listdir(input_path), colour='blue'):
            future = executor.submit(process_video, input_path, output_path, max_frames, file_name, target_width, target_height)
            futures.append(future)

        for future in tqdm(futures, colour='blue'):
            future.result()

max_frames = get_MaxFrame()
working_threads_pool(path, dest_path, max_frames, target_width, target_height)
