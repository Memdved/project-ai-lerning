import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# Параметры модели
img_width, img_height = 128, 128
batch_size = 32
epochs = 50 # Увеличиваем количество эпох
data_dir = 'dataSet'

# Создание DataFrame
paths = []
labels = []
for filename in os.listdir(data_dir):
  if filename.endswith(".jpg"):
    paths.append(os.path.join(data_dir, filename))
    # Извлечение имени пользователя из имени файла 
    user_name = filename.split('-')[1].split('.')[0]
    labels.append(user_name)

df = pd.DataFrame({'path': paths, 'label': labels})

# Разделение на тренировочный и тестовый наборы
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42) # 20% данных для теста

# Генератор данных с аугментацией
train_datagen = ImageDataGenerator(
  rescale=1./255,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  rotation_range=20, 
  width_shift_range=0.1, 
  height_shift_range=0.1,
  brightness_range=[0.2, 1.0] # Добавляем аугментацию яркости
)

test_datagen = ImageDataGenerator(rescale=1./255) # Только масштабирование для тестового набора

train_generator = train_datagen.flow_from_dataframe(
  dataframe=train_df,
  directory=None,
  x_col='path',
  y_col='label',
  target_size=(img_width, img_height),
  batch_size=batch_size,
  class_mode='categorical')

test_generator = test_datagen.flow_from_dataframe(
  dataframe=test_df,
  directory=None,
  x_col='path',
  y_col='label',
  target_size=(img_width, img_height),
  batch_size=batch_size,
  class_mode='categorical')

# Создание словаря для сопоставления индексов с именами
name_to_index = {name: index for index, name in enumerate(df['label'].unique())}
index_to_name = {index: name for name, index in name_to_index.items()}

# Более сложная модель с добавлением Dropout
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# Количество нейронов в выходном слое должно соответствовать количеству уникальных имен
unique_names = len(df['label'].unique())
model.add(Dense(unique_names, activation='softmax')) 

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
history = model.fit(
  train_generator,
  epochs=epochs,
  validation_data=test_generator) # Обучение с валидацией

# Сохранение модели
model.save('face_recognition_model.h5')

# Функция для распознавания лица
def recognize_face(image_path):
  img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
  img_array = tf.keras.preprocessing.image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.0

  prediction = model.predict(img_array)
  predicted_index = np.argmax(prediction)
  predicted_name = index_to_name[predicted_index] 
  return predicted_name

# Пример использования
image_path = 'test_images/face-645646.2.jpg' # Замените на ваш реальный путь
predicted_name = recognize_face(image_path)
print(f'Распознанное лицо: {predicted_name}')


with open('name_to_index.txt', 'w') as f:
  for name, index in name_to_index.items():
    f.write(f'{name},{index}\n')
