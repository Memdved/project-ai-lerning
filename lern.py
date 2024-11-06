import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l1


# Установка устройства для вычислений (GPU, если доступно)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
 tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Параметры модели
img_width, img_height = 128, 128
batch_size = 32
epochs = 50 
data_dir = 'dataSet'

# Создание DataFrame
paths = []
labels = []
for filename in os.listdir(data_dir):
 if filename.endswith(".jpg"):
  paths.append(os.path.join(data_dir, filename))
  user_name = filename.split('-')[1].split('.')[0]
  labels.append(user_name)

df = pd.DataFrame({'path': paths, 'label': labels})

# Разделение на тренировочный и тестовый наборы
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Генератор данных с аугментацией
train_datagen = ImageDataGenerator(
 rescale=1./255,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True,
 rotation_range=20, 
 width_shift_range=0.1, 
 height_shift_range=0.1,
 brightness_range=[0.2, 1.0] 
)

test_datagen = ImageDataGenerator(rescale=1./255) 

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

# Модель с большим количеством скрытых уровней
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
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
 validation_data=test_generator)

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
image_path = 'test_images/face-645646.2.jpg'
predicted_name = recognize_face(image_path)
print(f'Распознанное лицо: {predicted_name}')


with open('name_to_index.txt', 'w') as f:
 for name, index in name_to_index.items():
  f.write(f'{name},{index}\n')
  
# Пример использования предобученной модели VGG16 с регуляризацией L1
# Загрузить предобученную модель VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Заморозить первые слои
for layer in base_model.layers:
 layer.trainable = False

# Добавить новые слои
x = Flatten()(base_model.output)
x = Dense(512, activation='relu', kernel_regularizer=l1(0.01))(x) 
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l1(0.01))(x) 
x = Dropout(0.5)(x)
x = Dense(128, activation='relu', kernel_regularizer=l1(0.01))(x) 
x = Dropout(0.5)(x)
predictions = Dense(unique_names, activation='softmax')(x)

# Создать новую модель
model = Model(inputs=base_model.input, outputs=predictions)

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
history = model.fit(
 train_generator,
 epochs=epochs,
 validation_data=test_generator)

# Сохранение модели
model.save('face_recognition_model_vgg16.h5')
