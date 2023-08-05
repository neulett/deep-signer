import tkinter as tk
import customtkinter as ctk
from pynput.keyboard import Key, Controller
from pynput import keyboard

# Display define
root = ctk.CTk()
root.title("Deep Signer")
root.resizable(False, False)
ctk.set_appearance_mode('dark')

def on_press(key):
    try:
        print(key)
    except AttributeError:
        print(f"{key}")
    return key

def on_release(key):
    if key == keyboard.Key.esc:
        return False
    
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

detected_label = tk.Label(root, width=10, text=on_press)
detected_label.grid(row=1, column=1)

root.mainloop()