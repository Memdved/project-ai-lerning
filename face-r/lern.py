import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# пути к обучающим и тестовым наборам данных
train_data_path = "dataSet"
test_data_path = "testSet"

# параметры модели
input_shape = (200, 200, 3) # размер входного изображения

# создание набора данных
train_images = []
train_labels = []
test_images = []
test_labels = []

# Создание словаря user_id -> class_id
user_to_class = {}
next_class_id = 0

def get_class_id(user_name):
  global next_class_id # Объявляем next_class_id как глобальную переменную
  if user_name not in user_to_class:
    user_to_class[user_name] = next_class_id
    next_class_id += 1
  return user_to_class[user_name]



def process_dataset(data_path, images, labels):
  for root, dirs, files in os.walk(data_path):
    for file in files:
      try:
        user_name, frame_num, _ = file.split(".")
        user_name = user_name.split("-")[1] # Извлечение имени пользователя
        image = cv2.imread(os.path.join(root, file))
        if image is None:
          print(f"Warning: Could not load image: {os.path.join(root, file)}")
          continue
        image = cv2.resize(image, (200, 200))
        images.append(image)
        labels.append(get_class_id(user_name))
      except ValueError:
        print(f"Warning: Skipping file with unexpected format: {file}")

process_dataset(train_data_path, train_images, train_labels)
process_dataset(test_data_path, test_images, test_labels)


# нормализация пикселей в диапазон [0, 1]
train_images = np.array(train_images)
train_images = train_images / 255.0
test_images = np.array(test_images)
test_images = test_images / 255.0

# преобразование меток в one-hot кодировку
num_classes = len(user_to_class) #Обновляем num_classes после определения всех классов
train_labels = np.array(train_labels)
train_labels = to_categorical(train_labels, num_classes)
test_labels = np.array(test_labels)
test_labels = to_categorical(test_labels, num_classes)


# создание нейронной сети
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# разбиение набора данных на обучающий и проверочный
X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels, test_size=0.2)

# обучение модели
model.fit(X_train, y_train, epochs=10)

# оценка модели
score = model.evaluate(X_test, y_test)
print('Точность:', score[1])

# сохранение модели
model.save('model.h5')
