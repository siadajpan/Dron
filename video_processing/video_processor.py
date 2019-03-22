import cv2
import random
import math
import numpy as np

cap = cv2.VideoCapture('videos/Auta_z_drona_3.MP4')

while True:
    ret, frame = cap.read()
    frame = frame[::10, ::10, :]
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)

        corners = cv2.dilate(corners, None)

        frame[corners > 0.01 * corners.max()] = [0, 0, 255]

        cv2.imshow('t', frame)
        print(random.random())

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
