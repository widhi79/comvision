import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import os


class ModulDeteksiEmosi:
    def __init__(self):
        # Construct the absolute path to the model files
        module_dir = os.path.dirname(__file__)
        model_json_path = os.path.join(
            module_dir, "models", "emotion_model_structure.json"
        )
        model_weights_path = os.path.join(module_dir, "models", "emotion_model.h5")

        # Load the model architecture from JSON
        with open(model_json_path, "r") as json_file:
            model_json = json_file.read()
            self.emotion_model = model_from_json(model_json)

        # Load the model weights
        self.emotion_model.load_weights(model_weights_path)

        self.face_cascade = cv2.CascadeClassifier(
            os.path.join(module_dir, "models", "haarcascade_frontalface_default.xml")
        )

        self.emotions = [
            "Marah",
            "Jijik",
            "Takut",
            "Senang",
            "Sedih",
            "Terkejut",
            "Biasa",
        ]

    def detect_emotion(self, face_img):
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = np.reshape(face_img, (1, 48, 48, 1))
        face_img = face_img / 255.0

        emotion_pred = self.emotion_model.predict(face_img)
        emotion_index = np.argmax(emotion_pred)
        return self.emotions[emotion_index]

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        results = []
        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]
            emotion = self.detect_emotion(face)
            results.append((x, y, x + w, y + h, emotion))

        return results
