from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageTk
import math
from tkinter import filedialog,  Toplevel, Listbox, Text  # Импортируйте filedialog и Tk из tkinter
from interface import create_interface
from tkinter import  END
import os
import sqlite3
import shutil
import time
import threading
import json
from camera_handle import CameraHandle
from interface import open_archive
from anomalib.models import WinClip
from anomalib.engine import Engine

yolo_model = YOLO("best.onnx", task="detect")
winclip_engine = Engine(task="segmentation")
winclip_model = WinClip.load_from_checkpoint("winclip_model.ckpt")


is_continuous_infer = False

def create_database():
    '''Создание или подключение к базе данных SQLite'''
    conn = sqlite3.connect('images.db')
    c = conn.cursor()

    # Создание таблицы для изображений
    c.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL
        )
    ''')

    # Создание таблицы для результатов инференса
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            json_path TEXT NOT NULL,
            jpg_path TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    # Добавление демонстрационных данных
    # demo_json_path = 'result/result_20241215_173011.json'
    # demo_jpg_path = 'result/temp_image.jpg'
    # demo_timestamp = '2024-12-15 17:30:11'
    #
    # c.execute('''
    #           INSERT INTO results (json_path, jpg_path, timestamp)
    #           VALUES (?, ?, ?)
    #       ''', (demo_json_path, demo_jpg_path, demo_timestamp))
    # print("Демонстрационные данные успешно добавлены в таблицу results.")

    conn.commit()
    return conn


def capture_camera_periodically(panel2, image2, save_path, camera_index, for_nn=False):
    cam = CameraHandle().get_cam(camera_index)
    
    while True:
        ret, frame = cam.read()  # Чтение кадра
        if not ret:
            print("Не удалось захватить кадр.")
            break

        # Сохраняем изображение в папку tmpimg
        cv2.imwrite(save_path + "jopa.jpg", frame)
        if for_nn:
            CameraHandle.release_global_camera()
            return

        # Отображаем изображение в panel2
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.resize((panel2.winfo_width(), panel2.winfo_height()), Image.LANCZOS)

        image2["image"] = image
        img_tk = ImageTk.PhotoImage(image)
        panel2.config(image=img_tk)
        panel2.image = img_tk  # Сохраняем ссылку на изображение

        time.sleep(5)  # Задержка в 5 секунд

    CameraHandle().release_global_camera()


def get_image_names_from_db():
    '''Функция для получения имен изображений из базы данных'''
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('SELECT file_name FROM images')
    names = [row[0] for row in c.fetchall()]  # Получаем все имена файлов
    conn.close()
    return names


def load_image_from_db(selected_file_name, panel):
    '''Функция для загрузки изображения из базы данных'''
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


def generate_unique_filename_with_ext(directory, prefix, extension):
    '''
    Генерирует уникальное имя файла.
    directory: директория для сохранения файла.
    prefix: префикс имени файла.
    extension: расширение файла (например, ".jpg").
    '''
    timestamp = time.strftime("%Y%m%d_%H%M%S")  # Текущая дата и время
    counter = 1
    unique_name = f"{prefix}_{timestamp}{extension}"

    while os.path.exists(os.path.join(directory, unique_name)):
        unique_name = f"{prefix}_{timestamp}_{counter}{extension}"
        counter += 1

    return unique_name


def model_infer(image, save_directory, save_to_json=False) -> dict:
    '''
    Обрабатывает картинку image,
    сохраняет результат в текстовом виде и в виде результирующей картинки в save_directory,
    возвращает словарь Имя элемента -> количество штук на картинке
    '''
    results = yolo_model(image)
    res = results[0]

    elements = dict()
    for x in res.summary():
        name = x['name']
        if name not in elements.keys():
            elements[name] = 1
        else:
            elements[name] += 1

    # Генерация уникальных имён для JSON и JPG
    json_filename = generate_unique_filename_with_ext(save_directory, "result", ".json")
    jpg_filename = generate_unique_filename_with_ext(save_directory, "result", ".jpg")

    json_path = os.path.join(save_directory, json_filename)
    jpg_path = os.path.join(save_directory, jpg_filename)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Текущая дата и время

    # Сохранение JSON
    if save_to_json:
        to_json = results[0].to_json()  # Сохраняет в текстовом виде id, x, y, w, h
        with open(json_path, "w") as f:
            f.write(to_json)

    # Сохранение JPG
    results[0].save(jpg_path)  # Сохраняет картинку с наложенными лейблами

    # Сохранение путей и времени в базу данных
    conn = create_database()  # Получение подключения к базе данных
    c = conn.cursor()
    c.execute('INSERT INTO results (json_path, jpg_path, timestamp) VALUES (?, ?, ?)',
              (json_path, jpg_path, timestamp))
    conn.commit()
    conn.close()

    return elements



def update_interface_with_yolo(panel, img_path, save_path, elems_textbox, save_to_json=False):
    # Запускаем инференс
    elements = model_infer(img_path, save_path, save_to_json)

    elems_textbox.delete(1.0, END)  # Очищаем предыдущее содержимое

    # Выводим результаты в текстовое поле
    for key, value in elements.items():
        elems_textbox.insert(END, f"{key}: {str(value)}\n")  # Вставляем новые данные

    # Загрузка обработанного изображения и вывод на экран
    files = ["result/"+x for x in os.listdir("result/") if x.endswith(".jpg")] 
    last_modified_file = max(files, key=os.path.getmtime)

    img_with_labels = cv2.imread(last_modified_file)
    img_resized = cv2.resize(img_with_labels, (700, 600))  # Сжимаем изображение для отображения в panel2
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    panel.config(image=img_tk)
    panel.image = img_tk


def infer_image(panel, elems_textbox):
    # Используем изображение из image1 для инференса
    img1 = image1["image"]
    
    if img1 is None:
        print("Нет изображения для обработки")
        return

    save_path = "result"
    # Сохраняем изображение во временный файл для инференса
    temp_img_path = "temp_image.jpg"
    cv2.imwrite(temp_img_path, img1)  # Сохраняем img1 в файл

    update_interface_with_yolo(panel, temp_img_path, save_path, elems_textbox, save_to_json=True)
    


def resize_to_fit(img, target_width=640, target_height=480):
    '''Функция для сжатия изображения до 640x480, сохраняя пропорции'''
    #TODO: Почему до 640х480? Нейронка ест квадратные изображения. Для нейронки лучше 640х640
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
    '''Генерирует уникальное имя файла, если файл с таким именем уже существует.'''
    name, extension = os.path.splitext(original_name)
    unique_name = original_name
    counter = 1

    # Генерация нового имени, пока файл с таким именем существует
    while os.path.exists(os.path.join(directory, unique_name)):
        unique_name = f"{name}_{counter}{extension}"  # Пример: image_1.jpg
        counter += 1

    return unique_name


def load_image(panel, image_store):
    '''Функция для загрузки изображения'''
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


def load_second_image(panel,image_store):
    ''' Функция для загрузки второго изображения'''
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


def capture_from_camera(panel, image_store, camera_index):
    '''Функция для захвата 20 изображений с камеры и сохранения только последнего.'''
    cam = CameraHandle().get_cam(camera_index)

    last_frame = None  # Переменная для хранения последнего захваченного кадра
    interested_in_frame_num = 20

    for _ in range(interested_in_frame_num):  # Захватываем 20 изображений
        ret, frame = cam.read()
        if ret:
            last_frame = frame  # Сохраняем последний кадр
        else:
            print("Не удалось захватить кадр.")
            break

    CameraHandle().release_global_camera()

    if last_frame is not None:
        img_resized = resize_to_fit(last_frame)  # Сжимаем изображение до 640x480
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        panel.config(image=img_tk)
        panel.image = img_tk
        image_store["image"] = last_frame  # Сохраняем последнее изображение в переменной


def find_rotation_angle(image):
    '''Функция для нахождения угла поворота по самой длинной линии'''
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


def rotate_image(image, angle):
    '''Функция для поворота изображения на заданный угол'''
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Матрица поворота
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def find_differences(img1, img2):
    '''Функция для нахождения различий между двумя изображениями'''
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


def align_images(img1, img2):
    """Выровнять img2 относительно img1 с использованием ORB"""
    orb = cv2.ORB_create()

    # Обнаружение ключевых точек и дескрипторов
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Сопоставление точек
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Получение точек для вычисления матрицы преобразования
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Нахождение матрицы гомографии
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    height, width = img1.shape[:2]
    aligned_img = cv2.warpPerspective(img2, M, (width, height))
    return aligned_img


def rotate_image_button():
    '''Функция для поворота изображения второго изображения'''
    img2 = image2["image"]
    img1 = image1["image"]
    if img2 is not None:

        align_images(img1,img2)

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


def compare_images():
    '''Функция для нахождения и отображения различий между двумя изображениями'''
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


def continuous_infer_handler(root, panel, elem_textbox, camera_index):
    global is_continuous_infer

    is_continuous_infer = not is_continuous_infer

    def update_frame():
        # Read a frame from the camera
        ret, frame = cam.read()
        
        # If the frame was not captured successfully, break the loop
        if not ret:
            print("Error: Could not read frame.")
            return
        
        # Process the frame with the YOLO model
        results = yolo_model(frame)

        ''''''
        res = results[0]

        elements = dict()
        for x in res.summary():
            name = x['name']
            if name not in elements.keys():
                elements[name] = 1
            else:
                elements[name] += 1

        elem_textbox.delete(1.0, END)  # Очищаем предыдущее содержимое

        # Выводим результаты в текстовое поле
        for key, value in elements.items():
            elem_textbox.insert(END, f"{key}: {str(value)}\n")
        ''''''
        # Annotate the frame with detection results
        annotated_frame = results[0].plot()  # Get the annotated frame

        # Convert the frame to RGB format for displaying
        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # Update the panel with the new image
        panel.config(image=img_tk)
        panel.image = img_tk  # Keep a reference to avoid garbage collection

        # Call this function again after a short delay
        if is_continuous_infer:
            root.after(100, update_frame)  # Update every 100 ms


    if is_continuous_infer:
        cam = CameraHandle().get_cam(camera_index)
        update_frame()  # Start the frame update loop
    else:
        CameraHandle().release_global_camera()
        #print_jopa()


def diff_heatmap(panel):
    img1 = image1["image"]
    
    if img1 is None:
        print("Нет изображения для обработки")
        return

    img_to_infer = [img1]
    preds = winclip_engine.predict(winclip_model, dataloaders=img_to_infer)

    print(type(preds))



if __name__ == "__main__":    
    if not os.path.exists('data'):
        os.makedirs('data')
    create_database()
    # Переменные для хранения изображений
    image1 = {"image": None}
    image2 = {"image": None}

    # Создание интерфейса
    root, panel1, panel2, output_text, selected_camera = create_interface(
        load_image=lambda: load_image(panel1, image1),
        capture_from_camera=lambda: capture_from_camera(panel1, image1, selected_camera.get()),
        rotate_image_button=rotate_image_button,
        compare_images=compare_images,
        infer_image_with_yolo=lambda: infer_image(panel2, output_text), # Передача функции инференса
        continuous_infer=lambda: continuous_infer_handler(root, panel2, output_text, selected_camera.get()),
        load_second_image = lambda: load_second_image(panel2, image2),  # Передача функции загрузки второго изображения
        load_image_from_db=load_image_from_db,
        open_archive=open_archive,  # Передача функции для открытия архива
        diff_heatmap=lambda: diff_heatmap(panel2)
    )
    # Запуск основного цикла приложения
    root.mainloop()
