from tkinter import Tk, Button, Label, Frame, Text, Scrollbar, RIGHT, Y, END
from tkinter import ttk  # Импортируем ttk для комбобокса
import cv2  # Импортируем OpenCV для работы с камерами

def get_camera_list():
    # Получаем список доступных камер
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

def create_interface(load_image, capture_from_camera, rotate_image_button, compare_images, infer_image_with_yolo, load_second_image):
    root = Tk()
    root.title("Image Processing Application")
    root.geometry("800x800")  # Задайте размер окна по вашему желанию

    # Создаем фрейм для панели кнопок
    button_frame = Frame(root)
    button_frame.pack(side="right", fill="y")  # Размещаем панель справа и заполняем по вертикали
    # Создаем комбобокс для выбора камеры
    camera_list = get_camera_list()
    selected_camera = ttk.Combobox(button_frame, values=camera_list, state="readonly")
    selected_camera.pack(pady=5)

    if selected_camera.get() != "": 
        selected_camera.current(0)
    else:
        print("no cameras")

    # Создаем кнопку для загрузки изображения
    btn_load_image = Button(button_frame, text="Загрузить изображение", command=load_image)
    btn_load_image.pack(pady=5)  # Добавляем отступы для кнопок

    btn_load_second_image = Button(button_frame, text="Загрузить 2 изображение", command=load_second_image)
    btn_load_second_image.pack(pady=5)

    # Создаем кнопку для захвата изображения с камеры
    btn_capture_camera = Button(button_frame, text="Сделать фото с камеры", command=capture_from_camera)
    btn_capture_camera.pack(pady=5)

    # Создаем кнопку для поворота изображения
    btn_rotate_image = Button(button_frame, text="Повернуть изображение", command=rotate_image_button)
    btn_rotate_image.pack(pady=5)

    # Создаем кнопку для сравнения изображений
    btn_compare_images = Button(button_frame, text="Найти различия", command=compare_images)
    btn_compare_images.pack(pady=5)

    # Создаем кнопку для инференса YOLO
    btn_infer_image = Button(button_frame, text="Инференс YOLO", command=infer_image_with_yolo)
    btn_infer_image.pack(pady=5)

    # Создаем кнопку для выхода из приложения
    btn_exit = Button(button_frame, text="Выйти", command=root.quit)
    btn_exit.pack(pady=5)
    # Создаем текстовое поле для вывода данных
    output_text = Text(button_frame, height=10, width=30)
    output_text.pack(pady=5)

    # Создаем панели для отображения изображений
    panel1 = Label(root)  # Для первого изображения
    panel1.pack(side="left", padx=10, pady=10)

    panel2 = Label(root)  # Для второго изображения или различий
    panel2.pack(side="left", padx=10, pady=10)


    return root, panel1, panel2,output_text, selected_camera
