import cv2
import numpy as np
import tensorflow as tf

face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')
new_model = tf.keras.models.load_model('model/1')
categories = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def predict_expression(face):
    final_image = cv2.resize(face, (224, 224))
    final_image = np.expand_dims(final_image, axis = 0)
    final_image = final_image / 255.0

    predictions = new_model.predict(final_image)
    class_prediction = np.argmax(predictions)

    return categories[class_prediction]

def detect_face(frame):
    results = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]
        facess = face_cascade.detectMultiScale(roi_gray)

        if len(facess) == 0:
            continue

        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey : ey + eh, ex : ex + ew]

            result = predict_expression(face_roi)
            results.append(result)
    
    return results

def predict_image(images_path):
    final_resutls = {}

    for image in images_path:
        frame = cv2.imread(image)
        res = detect_face(frame)

        for r in res:
            if r not in final_resutls:
                final_resutls[r] = 0
            final_resutls[r] += 1
        
    return final_resutls