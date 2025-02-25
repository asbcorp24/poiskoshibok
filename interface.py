from tkinter import Tk, Label, Frame, Text,Toplevel, Listbox, Button
import ttkbootstrap as ttk  # Импортируем ttkbootstrap для улучшенных стилей
import cv2  # Импортируем OpenCV для работы с камерами

import sqlite3  # Импортируем sqlite3 для работы с базой данных
from PIL import Image, ImageTk  # Импортируем PIL для работы с изображениями
from mail import send_json_with_chart  # Импорт функции отправки почты
import sqlite3

import os
import time
import threading


def send_selected_result():
    """Открывает окно для выбора строки из results и отправки данных по почте."""
    send_window = Toplevel()
    send_window.title("Выбор строки для отправки")
    send_window.geometry("600x800")

    # Список записей
    listbox = Listbox(send_window, height=20, font=('Helvetica', 12))
    listbox.pack(fill="both", expand=True, padx=10, pady=10)

    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('SELECT id, json_path, jpg_path FROM results')
    records = c.fetchall()
    conn.close()

    for record in records:
        listbox.insert("end", f"ID: {record[0]} | JSON: {record[1]} | JPG: {record[2]}")

    def send_email_action():
        """Отправляет выбранную строку на почту."""
        selected = listbox.curselection()
        if not selected:
            print("Ничего не выбрано")
            return

        record_index = selected[0]
        record = records[record_index]

        json_path = record[1]
        jpg_path = record[2]

        if not os.path.exists(json_path) or not os.path.exists(jpg_path):
            print("Файлы не найдены")
            return

        try:
            send_json_with_chart(json_path, jpg_path)
        except Exception as e:
            print(f"Ошибка: {e}")

    send_button = Button(send_window, text="Отправить", command=send_email_action)
    send_button.pack(pady=10)



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


def open_archive():
    '''Открывает новое окно с архивом'''
    archive_window = Toplevel()  # Создаем новое окно
    archive_window.title("Архив")
    archive_window.geometry("800x600")

    # Список для отображения записей из базы данных
    listbox = Listbox(archive_window, height=20, font=('Helvetica', 12))
    listbox.pack(fill="both", expand=True, padx=10, pady=10)

    # Соединение с базой данных
    conn = sqlite3.connect('images.db')
    c = conn.cursor()
    c.execute('SELECT id, timestamp FROM results')
    records = c.fetchall()
    conn.close()

    # Добавление записей в список
    for record in records:
        listbox.insert("end", f"ID: {record[0]} | Время: {record[1]}")

    def load_selected_record(event):
        '''Загружает данные выбранной записи'''
        selected = listbox.curselection()
        if not selected:
            return

        record_index = selected[0]
        record_id = records[record_index][0]

        # Получение данных из базы данных
        conn = sqlite3.connect('images.db')
        c = conn.cursor()
        c.execute('SELECT json_path, jpg_path FROM results WHERE id = ?', (record_id,))
        result = c.fetchone()
        conn.close()

        if result:
            json_path, jpg_path = result

            # Загрузка и отображение изображения
            img = cv2.imread(jpg_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            img_label = Label(archive_window, image=img_tk)
            img_label.image = img_tk  # Сохраняем ссылку
            img_label.pack()

            # Загрузка и отображение содержимого JSON
            with open(json_path, "r") as f:
                json_content = f.read()

            json_label = Text(archive_window, height=10, wrap="word", font=('Helvetica', 12))
            json_label.insert("1.0", json_content)
            json_label.pack()

    # Обработчик выбора записи
    listbox.bind("<<ListboxSelect>>", load_selected_record)

def create_interface(load_image, capture_from_camera, rotate_image_button, compare_images, infer_image_with_yolo,
                     continuous_infer, load_second_image, load_image_from_db, open_archive, diff_heatmap):
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

    btn_cont_infer = ttk.Button(button_frame, text="Инференс в реальном времени", command=continuous_infer, **button_style)
    btn_cont_infer.pack(pady=10)

    # Кнопка для открытия архива
    btn_open_archive = ttk.Button(button_frame, text="Архив", command=open_archive, **button_style)
    btn_open_archive.pack(pady=10)

    btn_diff_heatmap = ttk.Button(button_frame, text="Карта различий", command=diff_heatmap, **button_style)
    btn_diff_heatmap.pack(pady=10)

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
    # Добавляем кнопку для отправки данных по почте
    btn_send_email = ttk.Button(button_frame, text="Отправить результат по почте", command=send_selected_result, **button_style)
    btn_send_email.pack(pady=10)

    return root, panel1, panel2, output_text, selected_camera