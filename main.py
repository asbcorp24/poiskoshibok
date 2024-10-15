import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


# Функция для сжатия изображения до 640x480, сохраняя пропорции
def resize_to_fit(img, target_width=640, target_height=480):
    h, w = img.shape[:2]
    aspect_ratio = w / h

    # Вычисляем новые размеры, сохраняя пропорции
    if aspect_ratio > 1:  # Ширина больше высоты
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:  # Высота больше или равна ширине
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


# Функция для загрузки изображения
def load_image(panel, image_var, title="Выберите изображение"):
    filename = filedialog.askopenfilename(title=title)
    if filename:
        img = cv2.imread(filename)
        img_resized = resize_to_fit(img)  # Сжимаем изображение до 640x480
        image_var["image"] = img_resized

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk


# Функция для нахождения различий между двумя изображениями
def find_differences(img1, img2):
    # Преобразуем изображения в серый цвет
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Находим абсолютную разницу между изображениями
    diff = cv2.absdiff(gray1, gray2)

    # Применяем пороговое значение для выделения различий
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Находим контуры различий
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Рисуем контуры на копии второго изображения
    img_with_differences = img2.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_differences, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_with_differences


# Функция для поворота изображения относительно самой длинной линии
def rotate_image():
    img = image1["image"]

    if img is not None:
        # Преобразуем изображение в черно-белое
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Используем метод Хафа для поиска линий
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        # Найдем самую длинную линию
        max_len = 0
        longest_line = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > max_len:
                max_len = length
                longest_line = (x1, y1, x2, y2)

        # Нарисуем самую длинную линию на изображении
        if longest_line is not None:
            x1, y1, x2, y2 = longest_line
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Вычисляем угол наклона самой длинной линии
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Поворачиваем изображение на этот угол
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))

        # Находим границы области изображения, чтобы сдвинуть его в левый верхний угол
        gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, thresh_rotated = cv2.threshold(gray_rotated, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_rotated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Найдем координаты области интереса
        x, y, w, h = cv2.boundingRect(contours[0])

        # Обрезаем и сдвигаем изображение в левый верхний угол
        cropped = rotated[y:y + h, x:x + w]

        # Сжимаем до 640x480 и обновляем изображение
        cropped_resized = resize_to_fit(cropped)
        img_rgb_rotated = cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2RGB)
        img_pil_rotated = Image.fromarray(img_rgb_rotated)
        img_tk_rotated = ImageTk.PhotoImage(image=img_pil_rotated)

        # Обновляем изображение в панели
        panel2.config(image=img_tk_rotated)
        panel2.image = img_tk_rotated


# Функция для нахождения и отображения различий между двумя изображениями
def compare_images():
    img1 = image1["image"]
    img2 = image2["image"]

    if img1 is not None and img2 is not None:
        img_with_differences = find_differences(img1, img2)
        img_rgb_diff = cv2.cvtColor(img_with_differences, cv2.COLOR_BGR2RGB)
        img_pil_diff = Image.fromarray(img_rgb_diff)
        img_tk_diff = ImageTk.PhotoImage(image=img_pil_diff)

        # Обновляем панель для отображения различий
        panel2.config(image=img_tk_diff)
        panel2.image = img_tk_diff


# Создание окна
root = Tk()
root.title("Поиск ошибок ")

# Панели для отображения изображений
panel1 = Label(root)
panel1.grid(row=0, column=0, padx=10, pady=10)
panel2 = Label(root)
panel2.grid(row=0, column=1, padx=10, pady=10)

# Переменные для хранения изображений
image1 = {"image": None}
image2 = {"image": None}

# Кнопки для загрузки изображений и сравнения
btn_load1 = Button(root, text="Загрузить первое изображение",
                   command=lambda: load_image(panel1, image1, "Выберите первое изображение"))
btn_load1.grid(row=1, column=0, padx=10, pady=10)

btn_load2 = Button(root, text="Загрузить второе изображение",
                   command=lambda: load_image(panel2, image2, "Выберите второе изображение"))
btn_load2.grid(row=1, column=1, padx=10, pady=10)

btn_compare = Button(root, text="Найти различия", command=compare_images)
btn_compare.grid(row=2, column=0, padx=10, pady=10)

btn_rotate = Button(root, text="Повернуть первое изображение", command=rotate_image)
btn_rotate.grid(row=2, column=1, padx=10, pady=10)

root.mainloop()
