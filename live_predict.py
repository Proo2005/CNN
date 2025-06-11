import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from preprocess import preprocess_face

# Load model and labels
model = load_model('../model/model.h5')
with open('../model/emotion_labels.pkl', 'rb') as f:
    emotion_labels = pickle.load(f)

# Load Haarcascade
face_cascade = cv2.CascadeClassifier('../haarcascade/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        face = preprocess_face(gray, x, y, w, h)
        pred = model.predict(face)
        label = emotion_labels[np.argmax(pred)]

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
