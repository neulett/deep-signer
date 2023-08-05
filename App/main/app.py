import cv2
import time
import clipboard
import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from pynput.keyboard import Key, Controller

# App Main
root = ctk.CTk()
root.title("Deep Signer")
root.resizable(False, False)
ctk.set_appearance_mode('dark')

# messagebox.showerror(title="Empty Clipboard", 
#             message="클립보드에 복사된 텍스트가 없습니다.")

# Functions
def draw_keypoints_from_npy(npy_path):
    frames = np.load(npy_path)
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

def ImageButton_Generator(img_path, target_size, target_frame, w, h, active_bg, bg):
    source = Image.open(img_path)
    source = source.resize(target_size)
    img = ImageTk.PhotoImage(source)
    btn = tk.Button(target_frame, image=img, relief='flat', width=w, height=h, activebackground=active_bg, bg=bg)
    btn.image = img
    return btn

def skeleton_on_canvas(image):  # draw keypoints on tkinter canvas
    pil_img = Image.fromarray(image)
    tk_img = ImageTk.PhotoImage(pil_img)   
    show_canvas.create_image(0, 0, anchor='nw', image=tk_img)
    show_canvas.image = tk_img   

def crop_clipboard():
    logs = []
    hkey_array = []
    result = clipboard.paste()  # get clipboard value
    hkey_array = result
    logs.extend(result)

class Keyboard_Listener:
    def __init__(self):
        self.keyboard = Controller()

        def keyboard_release(self, key):
            self.keyboard.press(key)
            self.keyboard.release(key)
            print(key)
            # return key

        def Input_HotKeys(self, key):
            with self.keyboard.pressed(Key.shift) and self.keyboard.pressed(Key.ctrl):
                input_val = keyboard_release(self, key) # call press & release function
                # detected_label.update(input_val)   # update detected text

# Frames
button_frame = ctk.CTkFrame(root)
button_frame.pack(side='left', expand=False, padx=20, pady=20)
frame = ctk.CTkFrame(root, corner_radius=5)
frame.pack(side='left', expand=False, padx=10, pady=20)

# Left Toolbar Button group
home_btn = ImageButton_Generator("icon/home2.png", (35, 35), button_frame, 35, 35, '#2b2b2b', '#2b2b2b')
home_btn.grid(row=0, column=0, padx=15, pady=40)
ttsl_btn = ImageButton_Generator("icon/gestures.png", (35, 35), button_frame, 35, 35, '#2b2b2b', '#2b2b2b')
ttsl_btn.grid(row=1, column=0, padx=15, pady=40)
setting_btn = ImageButton_Generator("icon/setting.png", (35, 35), button_frame, 35, 35, '#2b2b2b', '#2b2b2b')
setting_btn.grid(row=2, column=0, padx=15, pady=40)

# sub panel
detected_label = ctk.CTkLabel(frame, text='인식된 텍스트')    
detected_label.grid(row=0, column=0, padx=100, pady=240)

# canvas areas
show_canvas = ctk.CTkCanvas(root, width=350, height=480)
show_canvas.pack(side='left')

draw_keypoints_from_npy('./tk_program/KETI_SL_0000000001.npy')

root.mainloop()
