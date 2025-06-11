import cv2

def preprocess_face(gray_frame, x, y, w, h):
    face = gray_frame[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face.reshape(1, 48, 48, 1)
    face = face / 255.0
    return face
