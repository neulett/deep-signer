import os
import cv2
import threading
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


path = input("Source Path :")   # define paths
output_path = input("Dest Path :")

def extract_keypoints(filename, dest_path):
        mp_pose = mp.solutions.pose # define mediapipe pose model
        cap = cv2.VideoCapture(filename)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            pose_keypoints_list = []

            while True:
                opened, image = cap.read()
                if not opened:
                    break

                pose_results = pose.process(image)
                if pose_results.pose_landmarks:
                    pose_landmarks = pose_results.pose_landmarks
                    pose_keypoints = []
                    for landmark in pose_landmarks.landmark:
                        pose_keypoints.append([landmark.x, landmark.y, landmark.z])
                    pose_keypoints_list.append(pose_keypoints)

            pose_keypoints_array = np.array(pose_keypoints_list)
            
            np.save(f'{dest_path}/{os.path.splitext(os.path.basename(filename))[0]}_p.npy', pose_keypoints_array)

        cap.release()

def working_threads():
    executor = ThreadPoolExecutor(max_workers=4)
    futures = []

    for filename in tqdm(os.listdidr(path), colour='green'):
        future = executor.submit(extract_keypoints, filename, output_path)
        futures.append(future)

    print("File listup finish. Threading Start...")

    for future in tqdm(futures, colour='blue'):
        future.result()

    executor.shutdown()

working_threads()