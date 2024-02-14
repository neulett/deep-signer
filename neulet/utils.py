import os
import cv2
import math
import yaml
import numpy as np
import mediapipe as mp

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

class Utils:
    def __init__(self):
        self.keypoints = []
        self.keypoints_list = []
        opts = self.open_settings_yaml()
        self.WORKTYPE = opts['base_settings']['worktype']
        self.MODEL = opts['extract_settings']['model']
        self.HEIGHT = opts['extract_settings']['height']
        self.WIDTH = opts['extract_settings']['width']
        self.FPS = opts['extract_settings']['fps']
        self.WORKERS = opts['extract_settings']['workers']
        self.FEATURE_SAVE_PATH = opts['path']['keypoints_path']
        self.SOURCE_PATH = opts['path']['src_path']
        self.PADDED_SAVE_PATH = opts['path']['padded_path']
        
    @classmethod
    def open_settings_yaml(self, path='./command.yaml'):
        with open(path) as f:
            opts = yaml.load(f, Loader=yaml.FullLoader)
            return opts

    def extract_keypoints(self, filename):
        full_filename = os.path.join(self.PADDED_SAVE_PATH, filename)
        cap = cv2.VideoCapture(full_filename)

        if self.MODEL == "pose":
            solution = mp.solutions.pose.Pose
        elif self.MODEL == "holistic":
            solution = mp.solutions.holistic.Holistic
        else:
            raise ValueError("Invalid Model")
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

    def get_max_duration(self, filename):
        full_filename = os.path.join(self.SOURCE_PATH, filename)
        cap = cv2.VideoCapture(full_filename)
        duration = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT) // cap.get(cv2.CAP_PROP_FPS))

        if duration >= self.max_duration:
            self.max_duration = duration
            max_file = filename

        print(f"Max duration - {self.max_duration} sec - {max_file}")
        return self.max_duration
    
    def processing_padding_src(self, filename):
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

    def run_threads_ray(self):
        if self.WORKTYPE == 'padding':
            solution = self.processing_padding_src
        elif self.WORKTYPE == 'extract':
            solution = self.extract_keypoints
        else:
            raise ValueError("Invalid Worktype. please, check worktype in yaml.")

        for filename in tqdm(os.listdir(self.SOURCE_PATH), total=len(os.listdir(self.SOURCE_PATH))):
            solution(filename)

if __name__ == '__main__':
    instance = Utils()
    instance.run_threads_ray()
