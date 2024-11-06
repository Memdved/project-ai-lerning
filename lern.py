import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# Параметры модели
img_width, img_height = 128, 128
batch_size = 32
epochs = 10

# Путь к папке с данными
data_dir = 'dataSet'

# Создание DataFrame
paths = []
labels = []
for filename in os.listdir(data_dir):
 if filename.endswith(".jpg"):
  paths.append(os.path.join(data_dir, filename))
  user_id = int(filename.split('-')[1].split('.')[0])
  labels.append(user_id)

df = pd.DataFrame({'path': paths, 'label': [str(x) for x in labels]}) 

# Генератор данных
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_dataframe(
  dataframe=df,
  directory=None,
  x_col='path',
  y_col='label',
  target_size=(img_width, img_height),
  batch_size=batch_size,
  class_mode='categorical')

# Создание модели
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax')) # Задаем 3 нейрона, соответствующих 3 классам 

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
model.fit(
  train_generator,
  epochs=epochs)

# Сохранение модели
model.save('face_recognition_model.h5')

# Функция для распознавания лица (не изменена)
def recognize_face(image_path):
 img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
 img_array = tf.keras.preprocessing.image.img_to_array(img)
 img_array = np.expand_dims(img_array, axis=0)
 img_array /= 255.0

 prediction = model.predict(img_array)
 predicted_class = np.argmax(prediction)
 return predicted_class

# Пример использования
image_path = 'test_images/face-645646.2.jpg' # Замените на ваш реальный путь
predicted_class = recognize_face(image_path)
print(f'Распознанное лицо: {predicted_class}')
