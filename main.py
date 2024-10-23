from ultralytics import YOLO
import onnx
import onnxruntime
import cv2
import numpy as np
from PIL import Image, ImageTk
import math
from tkinter import filedialog, Tk  # Импортируйте filedialog и Tk из tkinter
from interface import create_interface
from tkinter import  END
import os
import sqlite3
import shutil
model = YOLO("best.onnx", task="detect")
if not os.path.exists('data'):
    os.makedirs('data')


# Создание или подключение к базе данных SQLite
def create_database():
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL
        )
    ''')
    conn.commit()
    return conn

# Функция для получения имен изображений из базы данных
def get_image_names_from_db():
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('SELECT file_name FROM images')
    names = [row[0] for row in c.fetchall()]  # Получаем все имена файлов
    conn.close()
    return names
# Функция для загрузки изображения из базы данных
def load_image_from_db(selected_file_name, panel):
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('SELECT file_path FROM images WHERE file_name = ?', (selected_file_name,))
    result = c.fetchone()
    conn.close()

    if result:
        file_path = result[0]
        img = cv2.imread(file_path)
        img_resized = resize_to_fit(img)  # Сжимаем изображение до 640x480
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk
        image1["image"] = img
    else:
        print("Изображение не найдено в базе данных.")

def save_image_to_db(conn, file_name, file_path):
    c = conn.cursor()
    c.execute('INSERT INTO images (file_name, file_path) VALUES (?, ?)', (file_name, file_path))
    conn.commit()
def model_infer(image,save_path,output_text) -> dict:
    '''
    Обрабатывает картинку img_path,
    сохраняет результат в текстовом виде и в виде результирующей картинки в save_path,
    возвращает словарь Имя элемента -> количество штук на картинке
    '''
    results = model(image)
    res = results[0]

    elements = dict()
    for x in res.boxes.data:  # Исправлено получение результатов
        name = x[-1]  # Последний элемент данных — это имя класса
        if name not in elements.keys():
            elements[name] = 1
        else:
            elements[name] += 1

    save_path = "result"
    results[0].save_txt(save_path + ".txt")  # сохраняет тхт в виде id x y w h
    results[0].save(save_path + ".jpg")  # сохраняет картинку с наложенными лейблами
    # Выводим результаты в текстовое поле
    output_text.delete(1.0, END)  # Очищаем предыдущее содержимое
    output_text.insert(END, str(elements))  # Вставляем новые данные
    return elements

def infer_image(panel,output_text):
    # Используем изображение из image1 для инференса
    img1 = image1["image"]
    if img1 is None:
        print("Нет изображения для обработки")
        return

    save_path = "result"
    # Сохраняем изображение во временный файл для инференса
    temp_img_path = "temp_image.jpg"
    cv2.imwrite(temp_img_path, img1)  # Сохраняем img1 в файл

    # Запускаем инференс
    model_infer(temp_img_path, save_path,output_text)

    # Загрузка обработанного изображения и вывод на экран
    img_with_labels = cv2.imread(save_path + ".jpg")
    img_resized = cv2.resize(img_with_labels, (700, 600))  # Сжимаем изображение для отображения в panel2
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    panel.config(image=img_tk)
    panel.image = img_tk


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
def generate_unique_filename(directory, original_name):
    """Генерирует уникальное имя файла, если файл с таким именем уже существует."""
    name, extension = os.path.splitext(original_name)
    unique_name = original_name
    counter = 1

    # Генерация нового имени, пока файл с таким именем существует
    while os.path.exists(os.path.join(directory, unique_name)):
        unique_name = f"{name}_{counter}{extension}"  # Пример: image_1.jpg
        counter += 1

    return unique_name
# Функция для загрузки изображения
def load_image(panel, image_store):
    filename = filedialog.askopenfilename(title="Выберите изображение")
    if filename:
        img = cv2.imread(filename)
        img_resized = resize_to_fit(img)  # Сжимаем изображение до 640x480
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk
        image_store["image"] = img  # Сохраняем изображение в переменной

        # Копируем изображение в папку data
        file_name = os.path.basename(filename)
        unique_file_name = generate_unique_filename('data', file_name)  # Получаем уникальное имя файла
        destination = os.path.join('data', unique_file_name)
        shutil.copy(filename, destination)  # Копируем файл в папку data

        # Сохраняем оригинальное имя и путь к файлу в базе данных
        conn = create_database()  # Подключаемся к базе данных
        save_image_to_db(conn, file_name, destination)  # Сохраняем оригинальное имя и путь
        conn.close()  # Закрываем соединение с базой данных
# Функция для загрузки второго изображения
def load_second_image(panel,image_store):
    filename = filedialog.askopenfilename(title="Выберите второе изображение")
    if filename:
        img = cv2.imread(filename)
        img_resized = resize_to_fit(img)  # Сжимаем изображение до 640x480
        image2["image"] = img_resized  # Сохраняем изображение во второй переменной

        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        panel.config(image=img_tk)
        panel.image = img_tk
        image_store["image"] = img
    # Функция для захвата изображения с камеры
def capture_from_camera(panel, image_store,camera_index):
    cap = cv2.VideoCapture(int(camera_index.split()[-1]))  # Получаем номер камеры
    ret, frame = cap.read()
    cap.release()  # Освобождаем камеру

    if ret:
        img_resized = resize_to_fit(frame)  # Сжимаем изображение до 640x480
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk
        image_store["image"] = frame  # Сохраняем изображение в переменной

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
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)  # Изменено пороговое значение

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Найдено контуров: {len(contours)}")  # Отладочный вывод

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

        # Сжимаем изображение с различиями до размеров панели
        img_resized_diff = resize_to_fit(img_with_differences, target_width=700, target_height=600)

        img_rgb_diff = cv2.cvtColor(img_resized_diff, cv2.COLOR_BGR2RGB)
        img_pil_diff = Image.fromarray(img_rgb_diff)
        img_tk_diff = ImageTk.PhotoImage(image=img_pil_diff)

        # Обновляем панель для отображения различий
        panel2.config(image=img_tk_diff)
        panel2.image = img_tk_diff


if __name__ == "__main__":
    # Переменные для хранения изображений
    image1 = {"image": None}
    image2 = {"image": None}

    # Создание интерфейса
    root, panel1, panel2, output_text, selected_camera = create_interface(
        load_image=lambda: load_image(panel1, image1),
        capture_from_camera=lambda: capture_from_camera(panel1, image1,selected_camera.get()),
        rotate_image_button=rotate_image_button,
        compare_images=compare_images,
        infer_image_with_yolo=lambda: infer_image(panel2, output_text), # Передача функции инференса
        load_second_image = lambda: load_second_image(panel2, image2),  # Передача функции загрузки второго изображения
        load_image_from_db=load_image_from_db
    )
    # Запуск основного цикла приложения
    root.mainloop()
