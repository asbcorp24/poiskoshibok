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

def create_interface(parent, load_image, capture_from_camera, rotate_image_button, compare_images, infer_image_with_yolo,
                     continuous_infer, load_second_image, load_image_from_db, open_archive, diff_heatmap, ocr):
    # Create the main frame for all content
    main_frame = ttk.Frame(parent)
    main_frame.pack(fill="both", expand=True)

    # Create a frame for the button panel on the right
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(side="right", fill="y", padx=10, pady=10)

    section_header = {"font": ("Helvetica", 12, "bold"), "anchor": "w"}

    ttk.Separator(button_frame, bootstyle="dark").pack(fill="x", pady=10)
    ttk.Label(button_frame, text="Камера и изображения", **section_header).pack(pady=(0, 5), anchor="w")

    # Camera selection
    camera_list = get_camera_list()
    selected_camera = ttk.Combobox(
        button_frame, 
        values=camera_list, 
        state="readonly",
        bootstyle="dark"
    )
    selected_camera.pack(fill="x", pady=5)
    if camera_list:
        selected_camera.current(0)

    # Buttons with rounded corners
    button_style = {"bootstyle": "outline", "width": 20}
    ttk.Button(button_frame, text="Загрузить изображение", command=load_image, **button_style).pack(pady=5)

    ttk.Button(button_frame, text="Загрузить 2 изображение", command=load_second_image,
                                       **button_style).pack(pady=5)

    ttk.Button(button_frame, text="Сделать фото с камеры", command=capture_from_camera,
                                    **button_style).pack(pady=5)

    # Add a combobox for loading images from the database
    ttk.Separator(button_frame, bootstyle="dark").pack(fill="x", pady=10)
    ttk.Label(button_frame, text="Открыть из датабазы", **section_header).pack(pady=(0, 5), anchor="w")
    
    image_names = get_image_names_from_db()
    # selected_image = ttk.Combobox(button_frame, values=image_names, state="readonly", font=('Helvetica', 12)).pack(pady=5)
    selected_image = ttk.Combobox(
        button_frame, 
        values=image_names, 
        state="readonly",
        bootstyle="dark"
    )
    selected_image.pack(fill="x", pady=5)

    # Button to load an image from the combobox
    ttk.Button(button_frame, text="Загрузить из базы данных",
                                  command=lambda: load_image_from_db(selected_image.get(), panel1), **button_style).pack(pady=5)

    # Second block: Image processing
    ttk.Label(button_frame, text="Обработка изображений", **section_header).pack(pady=(0, 5), anchor="w")
    ttk.Button(button_frame, text="Повернуть изображение", command=rotate_image_button,
                                  **button_style).pack(pady=5)

    ttk.Button(button_frame, text="Найти различия", command=compare_images, **button_style).pack(pady=5)

    ttk.Separator(button_frame, bootstyle="dark").pack(fill="x", pady=10)
    ttk.Label(button_frame, text="ИИ обработка", **section_header).pack(pady=(0, 5), anchor="w")
    ttk.Button(button_frame, text="Элементы фото", command=infer_image_with_yolo, **button_style).pack(pady=5)

    ttk.Button(button_frame, text="Элементы видео", command=continuous_infer, **button_style).pack(pady=5)

    # Button to open the archive
    ttk.Button(button_frame, text="Архив", command=open_archive, **button_style).pack(pady=5)

    ttk.Button(button_frame, text="Различия", command=diff_heatmap, **button_style).pack(pady=5)

    ttk.Button(button_frame, text="Найти текст", command=ocr, **button_style).pack(pady=5)

    ttk.Separator(button_frame, bootstyle="dark").pack(fill="x", pady=10)
    output_text = Text(
        main_frame, 
        height=5, 
        bg="#1B263B", 
        fg="white", 
        font=("Helvetica", 10),
        relief="flat",
        padx=5,
        pady=5
    )
    output_text.pack(side="bottom", fill="x", padx=10, pady=10)

    # Exit button at the bottom
    ttk.Button(
        button_frame, 
        text="Exit", 
        command=parent.quit,
        bootstyle="danger-outline",
        width=25,
        padding=5
    ).pack(side="bottom", pady=10)

    # Create a frame for displaying images on the left and right
    image_frame = ttk.Frame(main_frame)
    image_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

    # Panels for displaying images with borders
    panel1 = Label(image_frame, bg='#0D1B2A', bd=2, relief='solid')  # For the first image with a border
    panel1.pack(side="left", padx=10, pady=10, expand=True)

    panel2 = Label(image_frame, bg='#0D1B2A', bd=2, relief='solid', width=640, height=480)  # For the second image or differences with a border
    panel2.pack(side="right", padx=10, pady=10, expand=True)

    # Add a button to send data via email
    ttk.Button(button_frame, text="Отправить результат по почте", command=send_selected_result, **button_style).pack(pady=5)

    return panel1, panel2, output_text, selected_camera