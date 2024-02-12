import os
import cv2
import argparse
import numpy as np
import mediapipe as mp

from utils import Utils
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

class Engine:
    def __init__(self):
        self.keypoints = []
        self.keypoints_list = []
        opts = Utils.open_settings_yaml()
        self.MODEL = opts['setting']['model']
        self.HEIGHT = opts['setting']['height']
        self.WIDTH = opts['setting']['width']
        self.WORKERS = opts['setting']['workers']
        self.SOURCE_PATH = opts['setting']['src_path']
        self.FEATURE_SAVE_PATH = opts['setting']['keypoints_path']

    def extract_keypoints(self):
        for filename in tqdm(os.listdir(self.SOURCE_PATH), total=len(os.listdir(self.SOURCE_PATH))):
            full_filename = os.path.join(self.SOURCE_PATH, filename)
            cap = cv2.VideoCapture(full_filename)

            if self.MODEL == "pose":
                solution = mp.solutions.pose
            elif self.MODEL == "holistic":
                solution = mp.solutions.holistic
            else:
                raise ValueError("Invalid self.MODEL")

            with solution(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

                while True:
                    opened, image = cap.read()
                    if not opened:
                        break

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = pose.process(image_rgb)

                    if self.MODEL == "pose" and results.pose_landmarks:
                        for landmark in results.pose_landmarks.landmark:
                            self.keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                        self.keypoints_list.append(self.keypoints)

                        np.save(f"{self.FEATURE_SAVE_PATH}/{os.path.splitext(os.path.basename(filename))[0]}_pose.npy", self.keypoints_list) 

                    elif self.MODEL == "holistic" and (results.face_landmarks or
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
                        np.save(f"{self.FEATURE_SAVE_PATH}/{os.path.splitext(os.path.basename(filename))[0]}_holistic.npy", self.keypoints_list)

                cap.release()

    def run_threads(self):
        executor = ThreadPoolExecutor(max_workers=self.WORKERS)
        futures = []

        future = executor.submit(self.extract_keypoints, self.full_filename, self.FEATURE_SAVE_PATH)
        futures.append(future)

        for future in tqdm(futures, total=len(futures)):
            future.result()
        
if __name__ == '__main__':
    instance = Engine()
    instance.run_threads()