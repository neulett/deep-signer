import os
import cv2
import mediapipe as mp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

class Models:
    def __init__(self, source_path, dest_path):
        self.source_path = source_path
        self.dest_path = dest_path

    def extract_pose_keypoints(self, filename, dest_path):
        mp_pose = mp.solutions.pose
        cap = cv2.VideoCapture(filename)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            pose_keypoints_list = []

            while True:
                opened, image = cap.read()
                if not opened:
                    break

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(image_rgb)

                if pose_results.pose_landmarks:
                    pose_keypoints = []
                    for landmark in pose_results.pose_landmarks.landmark:
                        pose_keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                    pose_keypoints_list.append(pose_keypoints)

            np.save(f"{dest_path}/{os.path.splitext(os.path.basename(filename))[0]}_pose.npy", pose_keypoints_list)

        cap.release()

    def extract_holistic_keypoints(self, filename, dest_path):
        mp_holistic = mp.solutions.holistic
        cap = cv2.VideoCapture(filename)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            holistic_keypoints_list = []

            while True:
                opened, image = cap.read()
                if not opened:
                    break

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                holistic_results = holistic.process(image_rgb)

                if (holistic_results.face_landmarks or
                    holistic_results.left_hand_landmarks or
                    holistic_results.right_hand_landmarks or
                    holistic_results.pose_landmarks):

                    holistic_keypoints = {'face': [],
                                          'left_hand': [],
                                          'right_hand': [],
                                          'pose': []}

                    if holistic_results.pose_landmarks:
                        for landmark in holistic_results.pose_landmarks.landmark:
                            holistic_keypoints['pose'].append(
                                [landmark.x, landmark.y, landmark.z, landmark.visibility]
                            )

                    if holistic_results.face_landmarks:
                        for landmark in holistic_results.face_landmarks.landmark:
                            holistic_keypoints['face'].append(
                                [landmark.x, landmark.y, landmark.z, landmark.visibility]
                            )

                    if holistic_results.left_hand_landmarks:
                        for landmark in holistic_results.left_hand_landmarks.landmark:
                            holistic_keypoints['left_hand'].append(
                                [landmark.x, landmark.y, landmark.z, landmark.visibility]
                            )

                    if holistic_results.right_hand_landmarks:
                        for landmark in holistic_results.right_hand_landmarks.landmark:
                            holistic_keypoints['right_hand'].append(
                                [landmark.x, landmark.y, landmark.z, landmark.visibility]
                            )

                    holistic_keypoints_list.append(holistic_keypoints)

            np.save(f"{dest_path}/{os.path.splitext(os.path.basename(filename))[0]}_holistic.npy", holistic_keypoints_list)

        cap.release()

    def working_threads(self, model="pose"):
        executor = ThreadPoolExecutor(max_workers=4)
        futures = []

        if model == "pose":
            extract_method = self.extract_pose_keypoints
        elif model == "holistic":
            extract_method = self.extract_holistic_keypoints
        else:
            raise ValueError("Invalid model")

        for filename in tqdm(os.listdir(self.source_path), total=len(os.listdir(self.source_path))):
            full_filename = os.path.join(self.source_path, filename)
            future = executor.submit(extract_method, full_filename, self.dest_path)
            futures.append(future)

        print(f"{model} - File listup finish.. threads start..")

        for future in tqdm(futures, total=len(futures)):
            future.result()

        executor.shutdown()
