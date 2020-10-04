from cv2 import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_eye.xml")
cap = cv2.VideoCapture(0)

# implement recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

# read labels from .pickle file
labels = {}
with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
    # here we want labelname based on id
    # but our labels are stored like {name: id}
    # so we need to reverse that like {id: name}
    labels = {i: l for l, i in labels.items()}


while True:

    # capture frame by frame
    ret, frame = cap.read()

    # we have to convert the frame to being gray for cascade
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find faces in this gray frame
    faces = face_cascade.detectMultiScale(grayframe, scaleFactor=1.5, minNeighbors=5)

    # iterate through faces
    for (x, y, w, h) in faces:
        # print(x, y, w, h)

        # these values are region of interest ROI
        roi_gray = grayframe[y : y + h, x : x + w]

        # getting ROI for color frame
        roi_color = frame[y : y + h, x : x + w]

        # recognize using recognizer -- can also use any deep learning models
        id_, conf = recognizer.predict(roi_gray)  # conf = confidence level
        if conf >= 45 and conf <= 85:
            # print(id_)
            # here we are getting only ids but we need labels to these ids
            # so we read them from pickle file

            # read labels based on id
            print(labels[id_])

            # Put label on actual image
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 0, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        img_title = "4.png"
        # create image using ROI
        cv2.imwrite(img_title, roi_gray)

        # To draw a rectangle, we need color
        color = (255, 0, 0)  # BGR (blue, green, red)
        stroke = 2  # thickness of line
        endX = x + w
        endY = y + h
        cv2.rectangle(frame, (x, y), (endX, endY), color, stroke)
        # find eye
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # display the resulting frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release the capture
cap.release()
cv2.destroyAllWindows()