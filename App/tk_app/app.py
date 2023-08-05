import cv2
import time
import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from canvas_tools import Drawing, Generator

# App Main
root = ctk.CTk()
root.title("Deep Signer")
root.resizable(False, False)
ctk.set_appearance_mode('dark')

# Frames
button_frame = ctk.CTkFrame(root)
button_frame.pack(side='left', expand=False, padx=20, pady=20)
frame = ctk.CTkFrame(root, corner_radius=5)
frame.pack(side='left', expand=False, padx=10, pady=20)

# Left Toolbar Button group
home_btn = Generator.ImageButton_Generator("C:/Users/PCS/Desktop/deep_signer/App/icon/home.png", (35, 35), button_frame, 35, 35, '#2b2b2b', '#2b2b2b')
home_btn.grid(row=0, column=0, padx=15, pady=40)
ttsl_btn = Generator.ImageButton_Generator("C:/Users/PCS/Desktop/deep_signer/App/icon/sign-language.png", (35, 35), button_frame, 35, 35, '#2b2b2b', '#2b2b2b')
ttsl_btn.grid(row=1, column=0, padx=15, pady=40)
setting_btn = Generator.ImageButton_Generator("C:/Users/PCS/Desktop/deep_signer/App/icon/settings.png", (35, 35), button_frame, 35, 35, '#2b2b2b', '#2b2b2b')
setting_btn.grid(row=2, column=0, padx=15, pady=40)

# sub panel
detected_label = ctk.CTkLabel(frame, text='인식된 텍스트')    
detected_label.grid(row=0, column=0, padx=100, pady=240)

# canvas areas
show_canvas = ctk.CTkCanvas(root, width=350, height=480)
show_canvas.pack(side='left')

Drawing.draw_keypoints('./tk_program/KETI_SL_0000000001.npy')

root.mainloop()
