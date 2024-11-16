import cv2
import os


class FaceDetectorInterface:
  def detect_faces(self, image):
    raise NotImplementedError


class HaarCascadeFaceDetector(FaceDetectorInterface):
  def __init__(self, cascade_path="haarcascade_frontalface_default.xml"):
    self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)

  def detect_faces(self, image, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)):
    return self.detector.detectMultiScale(
      image, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize
    )


class FaceDatasetBuilder:
  def __init__(self, face_detector: FaceDetectorInterface, target_face_width=200, dataset_path="dataSet"):
    self.face_detector = face_detector
    self.target_face_width = target_face_width
    self.dataset_path = dataset_path

  def build_dataset(self, user_name, video_source=0, num_frames=1000):
    video = cv2.VideoCapture(video_source)
    frame_count = 0

    while ret:
      ret, frame = video.read()

      faces = self.face_detector.detect_faces(frame)
      for (x, y, w, h) in faces:
        frame_count += 1

        scale = self.target_face_width / w
        new_h = int(h * scale)
        face_x = x
        face_y = y
        face_w = self.target_face_width
        face_h = new_h

        face_x = max(0, face_x)
        face_y = max(0, face_y)
        face_w = min(face_w, frame.shape[1] - face_x)
        face_h = min(face_h, frame.shape[0] - face_y)

        self.save_face(frame, user_name, frame_count, face_x, face_y, face_w, face_h)

      if frame_count >= num_frames:
        break

    video.release()
    cv2.destroyAllWindows()

  def save_face(self, frame, user_name, frame_count, face_x, face_y, face_w, face_h):
    os.makedirs(self.dataset_path, exist_ok=True)
    face_image = frame[face_y:face_y + face_h, face_x:face_x + face_w]
    cv2.imwrite(f"{self.dataset_path}/face-{user_name}.{frame_count}.jpg", face_image)
    cv2.imshow("im", face_image)
    cv2.waitKey(100)


class FaceDatasetSaver:
  def __init__(self, dataset_path="dataSet"):
    self.dataset_path = dataset_path

  def save_face(self, frame, user_name, frame_count, face_x, face_y, face_w, face_h):
    os.makedirs(self.dataset_path, exist_ok=True)
    face_image = frame[face_y:face_y + face_h, face_x:face_x + face_w]
    cv2.imwrite(f"{self.dataset_path}/face-{user_name}.{frame_count}.jpg", face_image)
    cv2.imshow("im", face_image)
    cv2.waitKey(100)


class FaceDatasetBuilderImproved:
  def __init__(self, face_detector: FaceDetectorInterface, face_saver: FaceDatasetSaver, target_face_width=200):
    self.face_detector = face_detector
    self.target_face_width = target_face_width
    self.face_saver = face_saver

  def build_dataset(self, user_name, video_source=0, num_frames=1000):
    video = cv2.VideoCapture(video_source)
    frame_count = 0

    while True:
      ret, frame = video.read()
      if not ret:
        break

      faces = self.face_detector.detect_faces(frame)
      for (x, y, w, h) in faces:
        frame_count += 1

        scale = self.target_face_width / w
        new_h = int(h * scale)
        face_x = x
        face_y = y
        face_w = self.target_face_width
        face_h = new_h

        face_x = max(0, face_x)
        face_y = max(0, face_y)
        face_w = min(face_w, frame.shape[1] - face_x)
        face_h = min(face_h, frame.shape[0] - face_y)

        self.face_saver.save_face(frame, user_name, frame_count, face_x, face_y, face_w, face_h)

      if frame_count >= num_frames:
        break

    video.release()
    cv2.destroyAllWindows()

def main() -> None:
    user_name = input("Введите номер пользователя: ")

    face_detector = HaarCascadeFaceDetector()
    face_saver = FaceDatasetSaver()
    dataset_builder = FaceDatasetBuilderImproved(face_detector, face_saver)

    dataset_builder.build_dataset(user_name)



if __name__ == "__main__":
    main()
