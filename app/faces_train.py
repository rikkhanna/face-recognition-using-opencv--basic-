import os
import numpy as np
from PIL import Image
from cv2 import cv2
import pickle

# os.walk for image finding

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

# train opencv recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

current_id = 0  # id for image ROI
label_ids = {}
x_train = []
y_labels = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)

            # getting labels from directories
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            # print(label)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]  # id for each image ROI
            # print(label_ids)
            # every image has pixel value
            # train image into numpy array
            pil_image = Image.open(path).convert("L")  # convert to grayscale

            # Resize images for training
            size = (550, 550)  # create size
            finalImage = pil_image.resize(size, Image.ANTIALIAS)

            image_array = np.array(finalImage, "uint8")  # convert to numpy array
            # print(image_array)

            # finding ROI in numpy array i.e our training data
            faces = face_cascade.detectMultiScale(
                image_array, scaleFactor=1.5, minNeighbors=5
            )

            for (x, y, w, h) in faces:
                roi = image_array[y : y + h, x : x + w]
                x_train.append(roi)
                y_labels.append(id_)

# Using pickle to save label ids to be used in other modules
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)


# train opencv recognizer

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")

# TODO: Implement recognizer in faces.py