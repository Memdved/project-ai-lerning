import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

model = tf.keras.models.load_model('face_recognition_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img_width, img_height = 128, 128
offset = 50

video = cv2.VideoCapture(0)

def recognize_face(face_image):
  try:
    face_image = cv2.resize(face_image, (img_width, img_height))
    face_image = img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)
    face_image = face_image / 255.0

    prediction = model.predict(face_image)
    predicted_class = np.argmax(prediction)
    return predicted_class
  except:
    return "Неизвестный пользователь"

while True:
  ret, frame = video.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

  if len(faces) > 0:
    for (x, y, w, h) in faces:
      face_image = frame[y-offset:y+h+offset,x-offset:x+w+offset]

      if face_image.shape[0] > 0 and face_image.shape[1] > 0:
        predicted_class = recognize_face(face_image)

        cv2.rectangle(frame, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)

        if isinstance(predicted_class, str):
          cv2.putText(frame, predicted_class, (x, y - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (225, 0, 0), 2, cv2.LINE_AA)
        else:
          # Читаем словарь name_to_index из файла
          with open('name_to_index.txt', 'r') as f:
            name_to_index = {line.split(',')[0]: int(line.split(',')[1].strip()) for line in f}
          
          # Получаем имя пользователя из словаря по индексу
          predicted_name = list(name_to_index.keys())[predicted_class]
          
          cv2.putText(frame, predicted_name, (x, y - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (225, 0, 0), 2, cv2.LINE_AA) 

        cv2.imshow('Face', face_image)

  cv2.imshow('Video', frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video.release()
cv2.destroyAllWindows()
