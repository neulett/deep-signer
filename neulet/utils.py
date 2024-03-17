import os
import cv2
import ray
import math
import yaml
import numpy as np
import tkinter as tk
import mediapipe as mp
import clipboard as  cb
import customtkinter as ctk 
import win32clipboard as wcb
from tqdm.auto import tqdm
from PIL import Image, ImageTk
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
        self.PAD_KEYPOINTS_PATH = opts['path']['pad_keypoints_path']
        
    @classmethod
    def open_settings_yaml(self, path='./command.yaml'):
        with open(path) as f:
            opts = yaml.load(f, Loader=yaml.FullLoader)
            return opts

    @classmethod
    def get_clipboard_data(self):
        wcb.OpenClipboard()
        texts = wcb.GetClipboardData()
        wcb.CloseClipboard()
        return texts

    @classmethod
    def change_label(self, target_label):
        texts = ctk.StringVar()
        texts = self.get_clipboard_data()
        target_label.configure(text=texts)

    def get_max_video_duration(self, filename):
        full_filename = os.path.join(self.SOURCE_PATH, filename)
        cap = cv2.VideoCapture(full_filename)
        duration = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT) // cap.get(cv2.CAP_PROP_FPS))

        if duration >= self.max_duration:
            self.max_duration = duration
            max_file = filename

        print(f"Max duration - {self.max_duration} sec - {max_file}")
        return self.max_duration
    
    def get_max_array_duration(self, filename):
        max_height = 0
        max_width = 0

        array = np.load(os.path.join(self.keypoints_path, filename))
        height, width = array[:2]
        if height >= max_height:
            max_height = height
        elif width >= max_width:
            max_width = width

        print(f"Max_height : {max_height}\nMax_Width : {max_width}")
        return max_height, max_width

    @ray.remote
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
    
    @ray.remote
    def processing_padding_src(self, filename):
        self.max_duration = self.get_max_video_duration()
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

    @ray.remote
    def pad_array(self, filename):
        padded_arr = []
        max_height, max_width = self.get_max_array_duration()
        
        array = np.load(os.path.join(self.FEATURE_SAVE_PATH, filename))
        array_shape = array.shape
        height, width = array_shape.shape[:2]

        target_height = max_height - height
        target_width = max_width - width
        target_pad = ((0, target_height), (0, target_width), (0, 0))
        padded_array = np.pad(array, target_pad, mode='constant')
        padded_arr.append(padded_array)

        np.save(f"{self.PAD_KEYPOINTS_PATH}/{os.path.splitext(os.path.basename(filename))[0]}_pad.npy", padded_arr)
        del padded_arr

    def ray_runner(self, solution):
        ray.init(num_cpus=self.WORKERS)  
        futures = []

        if self.WORKTYPE == 'pad_video':
            solution = self.processing_padding_src
            paths_ = self.SOURCE_PATH
        elif self.WORKTYPE == 'extract':
            solution = self.extract_keypoints
            paths_ = self.PADDED_SAVE_PATH
        elif self.WORKTYPE == 'pad_array':
            solution = self.pad_array
            paths_ = self.FEATURE_SAVE_PATH
        else:
            raise ValueError("Invalid Worktype. please, check worktype in yaml.")

        for filename in os.listdir(paths_):
            futures.append(solution.remote(self, filename))

        ray.get(futures)  
        ray.shutdown()
        

class Drawing:
    def visualize_canvas(self, target_canvas, image):  # draw keypoints on tkinter canvas
        pil_img = Image.fromarray(image)
        tk_img = ImageTk.PhotoImage(pil_img)   
        target_canvas.create_image(0, 0, anchor='nw', image=tk_img)
        target_canvas.image = tk_img  

    def draw_keypoints(self, npy, root, canvas, img):
        self.visualize_canvas(canvas, img)
        frames = np.load(npy)
        for frame in frames:
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            for landmark in frame:
                x, y, _, visibility = landmark
                if visibility > 0.5:
                    cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])),
                               3, (0, 255, 0), thickness=-1)

                if frame[0][3] > 0.5 and frame[1][3] > 0.5:  # connect 1 - 0 pts
                    start_point = (int(frame[0][0] * image.shape[1]), int(frame[0][1] * image.shape[0]))
                    end_point = (int(frame[1][0] * image.shape[1]), int(frame[1][1] * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 0, 255), thickness=2)

                if frame[0][3] > 0.5 and frame[4][3] > 0.5:
                    start_point = (int(frame[0][0] * image.shape[1]), int(frame[0][1] * image.shape[0]))
                    end_point = (int(frame[4][0] * image.shape[1]), int(frame[4][1] * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 0, 255), thickness=2)

                if frame[1][3] > 0.5 and frame[2][3] > 0.5 and frame[3][3] > 0.5:
                    start_point = (int(frame[1][0] * image.shape[1]), int(frame[1][1] * image.shape[0]))
                    mid_point = (int(frame[2][0] * image.shape[1]), int(frame[2][1] * image.shape[0]))
                    mid_2_point = (int(frame[3][0] * image.shape[1]), int(frame[3][1] * image.shape[0]))
                    end_point = (int(frame[7][0] * image.shape[1]), int(frame[7][1] * image.shape[0]))
                    cv2.line(image, start_point, mid_point, (0, 0, 255), thickness=2)
                    cv2.line(image, mid_point, end_point, (0, 0, 255), thickness=2)

                if frame[4][3] > 0.5 and frame[5][3] > 0.5 and frame[6][3] > 0.5:
                    start_point = (int(frame[4][0] * image.shape[1]), int(frame[4][1] * image.shape[0]))
                    mid_point = (int(frame[5][0] * image.shape[1]), int(frame[5][1] * image.shape[0]))
                    mid_2_point = (int(frame[6][0] * image.shape[1]), int(frame[6][1] * image.shape[0]))
                    end_point = (int(frame[8][0] * image.shape[1]), int(frame[8][1] * image.shape[0]))
                    cv2.line(image, start_point, mid_point, (0, 0, 255), thickness=2)
                    cv2.line(image, mid_point, end_point, (0, 0, 255), thickness=2)

                if frame[10][3] > 0.5 and frame[9][3] > 0.5:
                    start_point = (int(frame[10][0] * image.shape[1]), int(frame[10][1] * image.shape[0]))
                    end_point = (int(frame[9][0] * image.shape[1]), int(frame[9][1] * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 0, 255), thickness=2)

                if frame[12][3] > 0.5 and frame[14][3] > 0.5:
                    start_point = (int(frame[12][0] * image.shape[1]), int(frame[12][1] * image.shape[0]))
                    end_point = (int(frame[14][0] * image.shape[1]), int(frame[14][1] * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 0, 255), thickness=2)

                if frame[11][3] > 0.5 and frame[23][3] > 0.5:
                    start_point = (int(frame[11][0] * image.shape[1]), int(frame[11][1] * image.shape[0]))
                    end_point = (int(frame[13][0] * image.shape[1]), int(frame[13][1] * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 0, 255), thickness=2)

                if frame[13][3] > 0.5 and frame[15][3] > 0.5:
                    start_point = (int(frame[13][0] * image.shape[1]), int(frame[13][1] * image.shape[0]))
                    end_point = (int(frame[15][0] * image.shape[1]), int(frame[15][1] * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 0, 255), thickness=2)

                if frame[12][3] > 0.5 and frame[11][3] > 0.5:
                    start_point = (int(frame[12][0] * image.shape[1]), int(frame[12][1] * image.shape[0]))
                    end_point = (int(frame[11][0] * image.shape[1]), int(frame[11][1] * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 0, 255), thickness=2)

                if frame[12][3] > 0.5 and frame[24][3] > 0.5:
                    start_point = (int(frame[12][0] * image.shape[1]), int(frame[12][1] * image.shape[0]))
                    end_point = (int(frame[24][0] * image.shape[1]), int(frame[24][1] * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 0, 255), thickness=2)

                if frame[11][3] > 0.5 and frame[23][3] > 0.5:
                    start_point = (int(frame[11][0] * image.shape[1]), int(frame[11][1] * image.shape[0]))
                    end_point = (int(frame[23][0] * image.shape[1]), int(frame[23][1] * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 0, 255), thickness=2)

                if frame[23][3] > 0.5 and frame[24][3] > 0.5:
                    start_point = (int(frame[23][0] * image.shape[1]), int(frame[23][1] * image.shape[0]))
                    end_point = (int(frame[24][0] * image.shape[1]), int(frame[24][1] * image.shape[0]))
                    cv2.line(image, start_point, end_point, (0, 0, 255), thickness=2)

            x_min, x_max = 150, 500
            y_min, y_max = 0, 480

            cropped_canvas = image[y_min:y_max, x_min:x_max]   

            self.visualize_canvas(cropped_canvas)
            root.update()
            root.after(30)

        cv2.destroyAllWindows()

if __name__ == '__main__':
    instance = Utils()
    instance.ray_runner()