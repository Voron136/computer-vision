import tensorflow as tf
from tensorflow.keras import layers, models
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Параметры
image_size = (64, 64)  # Размер изображения для обработки
batch_size = 32
epochs = 10
base_folder = r"C:\Users\Serval\Desktop\Sign-Language-Digits-Dataset-master\Dataset"
examples_folder = r"C:\Users\Serval\Desktop\Sign-Language-Digits-Dataset-master\Examples"

# 1. Подготовка данных с аугментацией
def load_data(base_folder):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,   # Масштабирование пикселей к диапазону [0, 1]
        validation_split=0.2,  # Разделение на тренировочные/валидационные данные
        rotation_range=15,     # Вращение изображений на ±15°
        width_shift_range=0.1, # Горизонтальный сдвиг до 10% от ширины
        height_shift_range=0.1, # Вертикальный сдвиг до 10% от высоты
        brightness_range=[0.8, 1.2], # Изменение яркости
        zoom_range=0.1         # Увеличение до 10%
    )
    
    train_data = datagen.flow_from_directory(
        base_folder,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )
    
    val_data = datagen.flow_from_directory(
        base_folder,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )
    
    return train_data, val_data

train_data, val_data = load_data(base_folder)

# 2. Создание модели с дополнительными слоями
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.BatchNormalization(),  # Нормализация
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),  # Нормализация
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),  # Нормализация
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),  # Нормализация
    layers.MaxPooling2D((2, 2)),
    
    layers.GlobalAveragePooling2D(),  
    layers.Dropout(0.5),  # Слой Dropout для предотвращения переобучения
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 классов для чисел от 0 до 9
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Обучение модели с визуализацией метрик
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# Построение графиков только для обучения
plt.figure(figsize=(12, 4))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# График потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='red')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 4. Сохранение модели 
model.save("gesture_recognition_model.keras")

# 5. Распознавание жестов
def recognize_gesture(folder_path, model):
    class_names = list(train_data.class_indices.keys())
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(img_path).resize(image_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  
            
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            print(f"File: {filename} -> Predicted Number: {class_names[predicted_class]}")

# Загрузка модели из нового формата
model = tf.keras.models.load_model("gesture_recognition_model.keras")

# Если требуется переобучение или оценка метрик
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Распознавание жестов
recognize_gesture(examples_folder, model)