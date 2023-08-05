from typing import Any
import cv2
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

class Drawing:
    def skeleton_on_canvas(target_canvas, image):  # draw keypoints on tkinter canvas
        pil_img = Image.fromarray(image)
        tk_img = ImageTk.PhotoImage(pil_img)   
        target_canvas.create_image(0, 0, anchor='nw', image=tk_img)
        target_canvas.image = tk_img  

    def draw_keypoints(self, npy, root, canvas, img):
        self.skeleton_on_canvas(canvas, img)
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

            skeleton_on_canvas(cropped_canvas)
            root.update()
            root.after(30)
        cv2.destroyAllWindows()

class Generator:
    def ImageButton_Generator(img_path, target_size, target_frame, w, h, active_bg, bg):
        source = Image.open(img_path)
        source = source.resize(target_size)
        img = ImageTk.PhotoImage(source)
        btn = tk.Button(target_frame, image=img, relief='flat', width=w, height=h, activebackground=active_bg, bg=bg)
        btn.image = img
        return btn