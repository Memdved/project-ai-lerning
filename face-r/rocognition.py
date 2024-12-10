import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# загрузка модели
model = load_model('model.h5')

# Загрузка каскада Хаара для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
  return faces

# инициализация веб-камеры
cap = cv2.VideoCapture(0)

while True:
  # чтение кадра с веб-камеры
  ret, frame = cap.read()
  if not ret:
    break

  # обнаружение лиц
  faces = detect_faces(frame)

  # распознавание лиц
  for (x, y, w, h) in faces:
    face = frame[y:y + h, x:x + w]
    face = cv2.resize(face, (200, 200))
    face = face / 255.0
    prediction = model.predict(np.array([face]))
    #Находим индекс с максимальной вероятностью
    predicted_class = np.argmax(prediction)
    #Необходимо преобразовать predicted_class обратно в имя пользователя
    #Это сложная задача, нужно использовать тот же словарь user_to_class, что и в lern.py
    #Или сохранять имя пользователя вместе с моделью

    #Временное решение для отображения номера класса
    cv2.putText(frame, f"User ID: {predicted_class}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

  # отображение кадра
  cv2.imshow('Распознавание лиц', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

