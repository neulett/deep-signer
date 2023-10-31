import os
import cv2
import argparse
import numpy as np
import mediapipe as mp

from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

class Models:
    def __init__(self):
        self.pose_keypoints = []
        self.pose_keypoints_list = []
        self.holistic_keypoints_list = []

    def extract_pose_keypoints(self):
        for filename in tqdm(os.listdir(SOURCE), total=len(os.listdir(SOURCE))):
            full_filename = os.path.join(SOURCE, filename)

            mp_pose = mp.solutions.pose
            cap = cv2.VideoCapture(full_filename)

            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

                while True:
                    opened, image = cap.read()
                    if not opened:
                        break

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(image_rgb)

                    if pose_results.pose_landmarks:
                        for landmark in pose_results.pose_landmarks.landmark:
                            self.pose_keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
                        self.pose_keypoints_list.append(self.pose_keypoints)

                np.save(f"{DEST}/{os.path.splitext(os.path.basename(filename))[0]}_pose.npy", self.pose_keypoints_list)

            cap.release()

    def extract_holistic_keypoints(self):
        for filename in tqdm(os.listdir(SOURCE), total=len(os.listdir(SOURCE))):
            full_filename = os.path.join(SOURCE, filename)
        
            mp_holistic = mp.solutions.holistic
            cap = cv2.VideoCapture(full_filename)

            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:


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

                        self.holistic_keypoints_list.append(holistic_keypoints)

            np.save(f"{DEST}/{os.path.splitext(os.path.basename(filename))[0]}_holistic.npy", self.holistic_keypoints_list)

        cap.release()

    def working_threads(self):
        if MODEL == "pose":
            self.extract_pose_keypoints()
        elif MODEL == "holistic":
            self.extract_holistic_keypoints()
        else:
            raise ValueError("Invalid Model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '--model', default='pose', type=str)
    parser.add_argument('-src', '--source', default='../video/padded_source', type=str)
    parser.add_argument('-dest', '--dest', default='../keypoints', type=str)
    # parser.add_argument('-workers', '--workers', default=4, type=int)
    args = parser.parse_args()

    MODEL = args.model
    SOURCE = args.source
    DEST = args.dest
    # WORKERS = args.workers

    instance = Models()
    instance.working_threads()
