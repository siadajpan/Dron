import cv2
import random
import math
import numpy as np

cap = cv2.VideoCapture('videos/Auta_z_drona_3.MP4')
haar_cascade = cv2.CascadeClassifier('car_models/car40_haar.xml')

while(cap.isOpened()):
    ret, frame = cap.read()
    print(ret)
    if ret:
        frame = frame[::10, ::10, :]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        #
        # corners = cv2.dilate(corners, None)
        #
        # frame[corners > 0.01 * corners.max()] = [0, 0, 255]
        #
        # cv2.imshow('t', frame)
        # print(random.random())

        cars = haar_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 0), 2)

        cv2.imshow('found', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
