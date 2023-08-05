import os
import cv2
import threading
import numpy as np
import mediapipe as mp
from tqdm import tqdm

path = '../pad/padded_source/'
output_path = '../keypoints/keypoints_save/'

def extract_keypoints(path):
    for file_name in tqdm(os.listdir(path), colour='green'):
        file_path = os.path.join(path, file_name)

        mp_pose = mp.solutions.pose # define mediapipe pose model
        cap = cv2.VideoCapture(file_path)

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
            np.save(f'./keypoints_save/{file_name[0:18]}.npy', pose_keypoints_array)

        cap.release()

def processing_with_threads(path):
    threads = []

    for file_name in tqdm(os.listdir(path)[:4], colour='green'):
        file_path = os.path.join(path, file_name)
        thread = threading.Thread(target=extract_keypoints, args=(file_path,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

processing_with_threads(path)