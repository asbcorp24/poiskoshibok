import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


# Функция для загрузки изображения
def load_image(panel, image_var, title="Выберите изображение"):
    filename = filedialog.askopenfilename(title=title)
    if filename:
        img = cv2.imread(filename)
        image_var["image"] = img
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk


# Функция для поиска различий между изображениями
def find_differences():
    img1 = image1["image"]
    img2 = image2["image"]

    if img1 is not None and img2 is not None:
        # Преобразуем изображения в черно-белые
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Поиск различий
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Увеличиваем разницу с помощью морфологических операций
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=3)

        # Находим контуры всех различий
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Рисуем все контуры на втором изображении
        img_diff = img2.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_diff, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Обновляем второе изображение с выделенными различиями
        img_rgb_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2RGB)
        img_pil_diff = Image.fromarray(img_rgb_diff)
        img_tk_diff = ImageTk.PhotoImage(image=img_pil_diff)
        panel2.config(image=img_tk_diff)
        panel2.image = img_tk_diff


# Создание окна
root = Tk()
root.title("Image Difference Finder")

# Панели для отображения изображений
panel1 = Label(root)
panel1.grid(row=0, column=0, padx=10, pady=10)
panel2 = Label(root)
panel2.grid(row=0, column=1, padx=10, pady=10)

# Переменные для хранения изображений
image1 = {"image": None}
image2 = {"image": None}

# Кнопки для загрузки изображений
btn_load1 = Button(root, text="Загрузить первое изображение",
                   command=lambda: load_image(panel1, image1, "Выберите первое изображение"))
btn_load1.grid(row=1, column=0, padx=10, pady=10)

btn_load2 = Button(root, text="Загрузить второе изображение",
                   command=lambda: load_image(panel2, image2, "Выберите второе изображение"))
btn_load2.grid(row=1, column=1, padx=10, pady=10)

# Кнопка для поиска различий
btn_find_diff = Button(root, text="Найти различия", command=find_differences)
btn_find_diff.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()
