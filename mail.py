import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import json

load_dotenv()  # Загружаем переменные окружения из .env

def send_email(subject, body, recipient_email, attachments):
    """Отправляет письмо с вложениями на указанный адрес."""
    sender_email = os.getenv('EMAIL_USER')
    sender_password = os.getenv('EMAIL_PASS')

    if not sender_email or not sender_password:
        raise ValueError("EMAIL_USER и EMAIL_PASS не настроены в .env")

    # Создаем сообщение
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Тело сообщения
    msg.attach(MIMEText(body, 'plain'))

    # Добавляем вложения
    for attachment in attachments:
        part = MIMEBase('application', 'octet-stream')
        with open(attachment, 'rb') as f:
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(attachment)}")
        msg.attach(part)

    # Подключаемся к SMTP-серверу и отправляем письмо
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())


def create_chart_from_json(json_path, save_path):
    """Создает график на основе JSON-файла."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Подсчет количества объектов каждого типа
    objects = {}
    for item in data:
        name = item['name']
        objects[name] = objects.get(name, 0) + 1

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.bar(objects.keys(), objects.values(), color='skyblue')
    plt.title('Object Distribution')
    plt.xlabel('Object Types')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Сохранение графика
    chart_path = os.path.join(save_path, 'chart.png')
    plt.savefig(chart_path)
    plt.close()
    return chart_path


def send_json_with_chart(json_path, jpg_path):
    """Отправляет JSON, изображение и график на почту."""
    recipient_email = os.getenv('EMAIL_RECIPIENT')
    if not recipient_email:
        raise ValueError("EMAIL_RECIPIENT не указан в .env")

    # Папка для сохранения графика
    save_path = os.path.dirname(json_path)
    chart_path = create_chart_from_json(json_path, save_path)

    try:
        send_email(
            subject="Результаты анализа микросхем с графиком",
            body="Во вложении JSON, изображение и график распределения объектов.",
            recipient_email=recipient_email,
            attachments=[json_path, jpg_path, chart_path]
        )
        print("Письмо успешно отправлено!")
    except Exception as e:
        print(f"Ошибка при отправке письма: {e}")
