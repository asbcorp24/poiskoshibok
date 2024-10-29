from tkinter import Tk, Label, Frame, Text
import ttkbootstrap as ttk  # Импортируем ttkbootstrap для улучшенных стилей
import cv2  # Импортируем OpenCV для работы с камерами

import sqlite3  # Импортируем sqlite3 для работы с базой данных
from PIL import Image, ImageTk  # Импортируем PIL для работы с изображениями

import os
import time
import threading

def get_camera_list():
    '''Получаем список доступных камер'''
    index = 0
    camera_list = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        camera_list.append(f"Камера {index}")
        index += 1
        cap.release()
    return camera_list


def get_image_names_from_db():
    '''Функция для получения имен изображений из базы данных'''
    image_names = []
    try:
        conn = sqlite3.connect('images.db')  # Подключение к базе данных
        cursor = conn.cursor()
        cursor.execute("SELECT file_name FROM images")  # Измените 'images' на имя вашей таблицы
        image_names = [row[0] for row in cursor.fetchall()]  # Получаем все имена изображений
        conn.close()
    except:
        print("Нет доступа к базе данных")
    return image_names




def create_interface(load_image, capture_from_camera, rotate_image_button, compare_images, infer_image_with_yolo,
                     load_second_image, load_image_from_db,start_camera_capture):
    root = ttk.Window(themename="darkly")  # Создаем окно с темной темой
    root.title("Image Processing Application")
    root.geometry("1000x800")  # Задаем размер окна

    # Создаем основной фрейм для всего контента
    main_frame = ttk.Frame(root)
    main_frame.pack(fill="both", expand=True)

    # Создаем фрейм для панели кнопок справа
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(side="right", fill="y", padx=10, pady=10)

    # Первый блок: Выбор камеры и загрузка изображений
    ttk.Label(button_frame, text="Выбор камеры и изображений").pack(pady=10)
    camera_list = get_camera_list()
    selected_camera = ttk.Combobox(button_frame, values=camera_list, state="readonly", font=('Helvetica', 12))
    selected_camera.pack(pady=10)
    if selected_camera.get() != "":
        selected_camera.current(0)
    else:
        print("no cameras")

    # Кнопки с закругленными углами
    button_style = {"bootstyle": "primary-outline", "width": 20}  # Используем стиль с закругленными углами
    btn_load_image = ttk.Button(button_frame, text="Загрузить изображение", command=load_image, **button_style)
    btn_load_image.pack(pady=10)

    btn_load_second_image = ttk.Button(button_frame, text="Загрузить 2 изображение", command=load_second_image,
                                       **button_style)
    btn_load_second_image.pack(pady=10)

    btn_capture_camera = ttk.Button(button_frame, text="Сделать фото с камеры", command=capture_from_camera,
                                    **button_style)
    btn_capture_camera.pack(pady=10)

    btn_capture_camera2 = ttk.Button(button_frame, text="Начать захват с камеры",
                                     command=start_camera_capture,
                                    **button_style)
    btn_capture_camera2.pack(pady=10)
    
    # Добавьте комбобокс для загрузки изображения из базы данных
    ttk.Label(button_frame, text="Выбор изображения из базы данных").pack(pady=10)
    image_names = get_image_names_from_db()  # Получите список имен изображений из базы данных

    selected_image = ttk.Combobox(button_frame, values=image_names, state="readonly", font=('Helvetica', 12))
    selected_image.pack(pady=10)

    # Кнопка для загрузки изображения из комбобокса
    btn_load_from_db = ttk.Button(button_frame, text="Загрузить из базы данных",
                                  command=lambda: load_image_from_db(selected_image.get(), panel1), **button_style)
    btn_load_from_db.pack(pady=10)
    # Второй блок: Обработка изображений
    ttk.Label(button_frame, text="Обработка изображений").pack(pady=10)
    btn_rotate_image = ttk.Button(button_frame, text="Повернуть изображение", command=rotate_image_button,
                                  **button_style)
    btn_rotate_image.pack(pady=10)

    btn_compare_images = ttk.Button(button_frame, text="Найти различия", command=compare_images, **button_style)
    btn_compare_images.pack(pady=10)

    btn_infer_image = ttk.Button(button_frame, text="Инференс YOLO", command=infer_image_with_yolo, **button_style)
    btn_infer_image.pack(pady=10)

    # Кнопка выхода внизу
    btn_exit = ttk.Button(button_frame, text="Выйти", command=root.quit, **button_style)
    btn_exit.pack(pady=10, side="bottom")

    # Создаем текстовое поле для вывода данных снизу
    output_text = Text(main_frame, height=5, width=30, bg='#1B263B', fg='white', font=('Helvetica', 12), bd=2,
                       relief='solid')
    output_text.pack(side="bottom", fill="x", padx=10, pady=10)

    # Создаем фрейм для отображения изображений слева и справа
    image_frame = ttk.Frame(main_frame)
    image_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    # Панели для отображения изображений с границами
    panel1 = Label(image_frame, bg='#0D1B2A', bd=2, relief='solid')  # Для первого изображения с рамкой
    panel1.pack(side="left", padx=10, pady=10, expand=True)

    panel2 = Label(image_frame, bg='#0D1B2A', bd=2, relief='solid', width=640,height=480)  # Для второго изображения или различий с рамкой
    panel2.pack(side="right", padx=10, pady=10, expand=True)

    return root, panel1, panel2, output_text, selected_camera