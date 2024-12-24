import tensorflow as tf
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

# Путь к обученной модели
model_path = r"C:\Users\Serval\Desktop\project\gesture_recognition_model.keras"
model = tf.keras.models.load_model(model_path)

# Список классов 
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
image_size = (64, 64)  # Размер изображения, использованный при обучении

def recognize_gesture(image_path):
    try:
        # Открытие и подготовка изображения
        img = Image.open(image_path).resize(image_size)
        img_array = np.array(img) / 255.0  # Нормализация
        img_array = np.expand_dims(img_array, axis=0)  
        
        # Предсказание
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        return class_names[predicted_class], confidence
    except Exception as e:
        return None, str(e)

def select_and_recognize_file():
    # Выбор файла через диалоговое окно
    file_path = filedialog.askopenfilename(
        title="Выберите файл изображения",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    
    if file_path:
        result, confidence = recognize_gesture(file_path)
        if result:
            messagebox.showinfo("Результат", f"Распознанное число: {result}\nУверенность: {confidence:.2%}")
        else:
            messagebox.showerror("Ошибка", f"Ошибка обработки файла: {confidence}")

# Интерфейс пользователя
root = tk.Tk()
root.title("Распознавание жестов")
root.geometry("300x150")

label = tk.Label(root, text="Выберите файл для распознавания")
label.pack(pady=10)

button = tk.Button(root, text="Выбрать файл", command=select_and_recognize_file)
button.pack(pady=10)

root.mainloop()

