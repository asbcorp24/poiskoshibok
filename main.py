import cv2
import numpy as np
from PIL import Image, ImageTk
import math
from tkinter import filedialog, Tk  # Импортируйте filedialog и Tk из tkinter
from interface import create_interface
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
def load_image(panel):
    filename = filedialog.askopenfilename(title="Выберите изображение")
    if filename:
        img = cv2.imread(filename)
        img_resized = resize_to_fit(img)  # Сжимаем изображение до 640x480
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk

# Функция для захвата изображения с камеры
def capture_from_camera(panel):
    cap = cv2.VideoCapture(0)  # Открываем камеру (0 — это индекс камеры по умолчанию)
    ret, frame = cap.read()
    cap.release()  # Освобождаем камеру

    if ret:
        img_resized = resize_to_fit(frame)  # Сжимаем изображение до 640x480
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk

# Функция для нахождения угла поворота по самой длинной линии
def find_rotation_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Поиск линий с использованием преобразования Хафа
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    if lines is not None:
        # Найдем самую длинную линию
        max_len = 0
        best_line = None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > max_len:
                max_len = length
                best_line = (x1, y1, x2, y2)

        if best_line:
            x1, y1, x2, y2 = best_line
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            return angle, best_line
    return 0, None

# Функция для поворота изображения на заданный угол
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Матрица поворота
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

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

# Функция для поворота изображения второго изображения
def rotate_image_button():
    img2 = image2["image"]

    if img2 is not None:
        # Поворот второго изображения по наибольшей линии
        angle, best_line = find_rotation_angle(img2)
        if best_line:
            cv2.line(img2, (best_line[0], best_line[1]), (best_line[2], best_line[3]), (0, 0, 255), 2)
            img2_rotated = rotate_image(img2, -angle)
            image2["image"] = img2_rotated

            img_rgb = cv2.cvtColor(img2_rotated, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            panel2.config(image=img_tk)
            panel2.image = img_tk

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

# Переменные для хранения изображений
image1 = {"image": None}
image2 = {"image": None}

# Создание интерфейса
root, panel1, panel2 = create_interface(load_image, capture_from_camera, rotate_image_button, compare_images)

# Запуск основного цикла приложения
root.mainloop()
