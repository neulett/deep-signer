import os
import cv2
import math
import yaml
import numpy as np
import mediapipe as mp

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

class utils:
    def __init__(self):
        self.keypoints = []
        self.keypoints_list = []

    def extract_keypoints(self):
        for filename in tqdm(os.listdir(SOURCE), total=len(os.listdir(SOURCE))):
            full_filename = os.path.join(SOURCE, filename)
            cap = cv2.VideoCapture(full_filename)

            if model == "pose":
                solution = mp.solutions.pose
            elif model == "holistic":
                solution = mp.solutions.holistic
            else:
                raise ValueError("Invalid Model")

            with solution(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

                while True:
                    opened, image = cap.read()
                    if not opened:
                        break

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)

                    if model == "pose" and results.pose_landmarks:
                        for landmark in results.pose_landmarks.landmark:
                            self.keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                        self.keypoints_list.append(self.keypoints)

                        np.save(f"{DEST}/{os.path.splitext(os.path.basename(filename))[0]}_pose.npy", self.keypoints_list) 

                    elif model == "holistic" and (results.face_landmarks or
                                                  results.left_hand_landmarks or
                                                  results.right_hand_landmarks or
                                                  results.pose_landmarks):
                        keypoints = {'face': [],
                                     'left_hand': [],
                                     'right': [],
                                     'pose': []}
                        
                        if results.pose_landmarks:
                            for landmark in results.pose_landmarks.landmark:
                                keypoints['pose'].append([
                                    landmark.x, landmark.y, landmark.z, landmark.visibility
                                ])

                        if results.face_landmarks:
                            for landmark in results.face_landmarks.landmark:
                                keypoints['face'].append([
                                    landmark.x, landmark.y, landmark.z, landmark.visibility
                                ])

                        if results.left_hand_landmarks:
                            for landmark in results.left_hand_landmarks.landmark:
                                keypoints['left_hand'].append([
                                    landmark.x, landmark.y, landmark.z, landmark.visibility
                                ])

                        if results.right_hand_landmarks:
                            for landmark in results.right_hand_landmarks.landmark:
                                keypoints['right_hand'].append([
                                    landmark.x, landmark.y, landmark.z, landmark.visibility
                                ])

                        self.keypoints_list.append(keypoints)
                        np.save(f"{DEST}/{os.path.splitext(os.path.basename(filename))[0]}_holistic.npy", self.keypoints_list)

                cap.release()