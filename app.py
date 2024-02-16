import cv2
import time
import keyboard
import numpy as np
import tkinter as tk
import customtkinter as ctk

from tkinter import *
from neulet import utils
from tkinter import messagebox
from PIL import Image, ImageTk

Utils = utils.Utils()
Drawing = utils.Drawing()

def ImageButton_Generator(img_path, target_size, target_frame, w, h, active_bg, bg):
    source = Image.open(img_path)
    source = source.resize(target_size)
    img = ImageTk.PhotoImage(source)
    btn = tk.Button(target_frame, image=img, relief='flat', width=w, height=h, activebackground=active_bg, bg=bg)
    btn.image = img
    return btn

def keyboard_event_handler():
    keyboard.add_hotkey('ctrl+c', Utils.change_label_text(detected_label))
    root.after(1000, keyboard_event_handler)

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
home_btn = ImageButton_Generator('../assets/home.png', (35, 35), button_frame, 35, 35, '#2b2b2b', '#2b2b2b')
home_btn.grid(row=0, column=0, padx=15, pady=40)
ttsl_btn = ImageButton_Generator('../assets/sign-language.png', (35, 35), button_frame, 35, 35, '#2b2b2b', '#2b2b2b')
ttsl_btn.grid(row=1, column=0, padx=15, pady=40)
setting_btn = ImageButton_Generator('../assets/settings.png', (35, 35), button_frame, 35, 35, '#2b2b2b', '#2b2b2b')
setting_btn.grid(row=2, column=0, padx=15, pady=40)

# sub panel
detected_label = ctk.CTkLabel(frame, text="입력된 텍스트")
detected_label.grid(row=0, column=0, padx=100, pady=240)
        
# canvas areas
show_canvas = ctk.CTkCanvas(root, width=400, height=630)
show_canvas.pack(side='left')

keyboard_event_handler()
root.mainloop()
