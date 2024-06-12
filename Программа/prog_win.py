import keras
from customtkinter import *
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import numpy as np


class App:
    def __init__(self, window):
        self.window = window

        self.img_path = None
        self.img_tk = None

        self.zoomcycle = 0
        self.zimg_id = None

        self.init_ui()


    def init_ui(self):
        # Надпись
        self.label = CTkLabel(self.window, text="Нажмите на кнопку, чтобы загрузить изображение и сделать прогноз", font=("Arial", 14))
        self.label.pack(padx=50, pady=5)

        # Кнопка загрузки
        self.load_button = CTkButton(self.window, text="Загрузить", command=self.load_image, font=("Arial", 14))
        self.load_button.pack(padx=5, pady=5)

    def load_image(self):
        self.img_path = tk.filedialog.askopenfilename(title="Выберите изображение",
                                                      filetypes=[("Изображения", "*.jpg *.png *.jpeg")])

        if self.img_path:
            self.clear_frame()
            self.show_image()
            self.show_diagnosis()

    def clear_frame(self):
        for child in self.window.winfo_children():
            if child != self.label and child != self.load_button:
                child.destroy()

    def show_image(self):
        img_frame = CTkFrame(self.window)
        img_frame.pack()
        self.canvas = CTkCanvas(width=800, height=800, borderwidth=0, highlightthickness=0)
        self.canvas.pack()

        self.window.bind("<MouseWheel>", self.zoomer)
        self.canvas.bind("<Motion>", self.crop)


        path_label = CTkLabel(img_frame, text="")
        path_label.pack()

        img = self.convert_to_png()
        img = img.resize((875, 875), Image.LANCZOS)
        self.orig_img = img.copy()  # Keep a copy of the original image
        self.img_tk = ImageTk.PhotoImage(img)  # Initial image
        # img_label.configure(image=self.img_tk)

        self.canvas.create_image(0, 0, image=self.img_tk, anchor="nw")
        path_label.configure(text=f'Путь: {self.img_path}')

    def convert_to_png(self):

        file_extension = self.img_path.split('.')[-1].lower()

        if file_extension in ['jpg', 'jpeg']:
            image = Image.open(self.img_path)
        elif file_extension in ['png']:
            image = Image.open(self.img_path)
            image = image.convert('L')
        return image

    def show_diagnosis(self):

        model_pn = keras.models.load_model('model_pneumonia_15.h5')
        model_tb = keras.models.load_model('model_tuberculosis_09.h5')

        # Preprocess the image for the model (resize, normalize, etc.)
        img = keras.preprocessing.image.load_img(self.img_path, target_size=(512, 512), color_mode='grayscale')  # Adjust size based on model
        img = keras.preprocessing.image.img_to_array(img)
        img = img / 255.0  # Normalize pixel values between 0 and 1
        img = np.expand_dims(img, axis=0)  # Add a batch dimension

        # Make prediction using the model
        prediction_pn = model_pn.predict(img)
        pneumonia_probability = prediction_pn[0][0] * 100  # Assuming first element is pneumonia probability

        prediction_tb = model_tb.predict(img)
        tuberculosis_probability = prediction_tb[0][0] * 100

        # Update the diagnosis label with the predicted probability
        predict_label = CTkLabel(self.window, text= "Прогноз:", pady=10, font=("Arial", 14))
        diagnosis_pn = CTkLabel(self.window, text= f"Пневмония с вероятностью {pneumonia_probability:.2f}%", font=("Arial", 14))
        diagnosis_tb = CTkLabel(self.window, text= f"Туберкулёз с вероятностью {tuberculosis_probability:.2f}%", font=("Arial", 14))
        predict_label.pack()
        diagnosis_pn.pack()
        diagnosis_tb.pack()

    def zoomer(self, event):
        if (event.delta > 0):
            if self.zoomcycle != 3: self.zoomcycle += 1
        elif (event.delta < 0):
            if self.zoomcycle != 0: self.zoomcycle -= 1
        self.crop(event)

    def crop(self, event):
        if self.zimg_id: self.canvas.delete(self.zimg_id)
        if (self.zoomcycle) != 0:
            x, y = event.x, event.y
            if self.zoomcycle == 1:
                tmp = self.orig_img.crop((x - 65, y - 65, x + 65, y + 65))
            elif self.zoomcycle == 2:
                tmp = self.orig_img.crop((x - 45, y - 45, x + 45, y + 45))
            elif self.zoomcycle == 3:
                tmp = self.orig_img.crop((x - 30, y - 30, x + 30, y + 30))
            size = 400, 400
            self.zimg = ImageTk.PhotoImage(tmp.resize(size))
            self.zimg_id = self.canvas.create_image(event.x, event.y, image=self.zimg)


if __name__ == "__main__":
    window = CTk()
    window.title("Classification diseases")
    App = App(window)
    window.mainloop()
