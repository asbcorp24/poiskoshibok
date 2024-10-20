from tkinter import *


def create_interface(load_image, capture_from_camera, rotate_image_button, compare_images):
    root = Tk()
    root.title("Image Comparison with Rotation")
    root.geometry("1000x838")  # Установка размеров окна приложения

    # Панели для отображения изображений
    panel1 = Label(root,  bg="blue")  # Синий прямоугольник вместо первого изображения
    panel1.place(x=1000-301, y=10,width=300, height=300)  # Позиция первого изображения (справа вверху)

    panel2 = Label(root,  bg="blue")  # Синий прямоугольник для второго изображения
    panel2.place(x=10, y=10,width=680, height=600,)  # Позиция второго изображения (слева 10, сверху 10)

    button_panel = Frame(root)
    button_panel.place(x=1000-301, y=310, width=183, height=600)  # Позиция и размеры панели кнопок

    # Кнопки для загрузки изображений и сравнения
    btn_load1 = Button(button_panel, text="Загрузить первое изображение",
                       command=lambda: load_image(panel1))
    btn_load1.pack(pady=5)  # Использование pack для кнопок

    btn_load2_file = Button(button_panel, text="Загрузить второе изображение",
                            command=lambda: load_image(panel2))
    btn_load2_file.pack(pady=5)

    btn_capture2 = Button(button_panel, text="Сделать снимок с камеры", command=lambda: capture_from_camera(panel2))
    btn_capture2.pack(pady=5)

    btn_rotate = Button(button_panel, text="Повернуть изображение", command=rotate_image_button)
    btn_rotate.pack(pady=5)

    btn_compare = Button(button_panel, text="Найти различия", command=compare_images)
    btn_compare.pack(pady=5)

    return root, panel1, panel2
