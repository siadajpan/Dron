import numpy as np
import cv2
from sim_utils.classes import Car, Display, Driver, Drone, ImageProcessor, DroneParametersReader
import random
import threading

if __name__ == '__main__':
    field = cv2.imread('images/grass.jpg')
    field_size = field.shape[:2]

    car1_im = cv2.imread('images/car1s.png')
    car2_im = cv2.imread('images/car2s.png')
    car3_im = cv2.imread('images/car3s.png')

    haar_cascade = cv2.CascadeClassifier('car_models/git_cars/2.xml')

    cars = [Car() for i in range(3)]

    drivers = [Driver(car, field_size) for car in cars]
    display = Display(field, [car1_im, car2_im, car3_im], cars)

    hor_angle = 1.13
    screen_ratio = 0.6
    pitch = 0
    yaw = 0
    altitude = 100  # in decimeter
    cam_settings = hor_angle, screen_ratio
    drone = Drone(cam_settings, altitude, pitch, yaw)
    print('init drone position to:', cars[0].get_position)
    drone.init_position(cars[0].get_position)

    processor = ImageProcessor((hor_angle, screen_ratio), 1)

    # this daemon is reading drone parameters (gps position, altitude, pitch, yaw) every second
    drone_updater = DroneParametersReader(drone, processor)
    drone_updater.setDaemon(True)
    drone_updater.start()

    dt = 0.02

    while True:
        for driver in drivers:
            driver.drive(dt)

        display.update_field()
        drone_input = drone.display_visible(display.field_with_cars)

        # show them on the screen
        # drone_input[] = [0, 0, 255]

        display.show_field()
        cv2.imshow('corners', drone_input)

        flat_image = processor.transform_view(drone_input)

        # cars = processor.find_cars_haar(flat_image, haar_cascade)
        cars = processor.find_cars(flat_image)

        if cars:
            for (x, y, w, h) in cars:
                cv2.rectangle(flat_image, (x, y), (x + w, y + h), (150, 150, 0), 2)
        # else:
        #     break
        cv2.imshow('found', flat_image)
        # cv2.imshow('transfered', processor.find_cars(flat_image))

        key = cv2.waitKey(int(dt * 1000))
        if key == ord('q'):
            break
        elif key == ord('z'):
            drone.increase_altitude()
        elif key == ord('x'):
            drone.decrease_altitude()
        elif key == ord('w'):
            drone.move('up')
        elif key == ord('a'):
            drone.move('left')
        elif key == ord('s'):
            drone.move('down')
        elif key == ord('d'):
            drone.move('right')
        elif key == ord('r'):
            drone.rotate_pitch('up')
        elif key == ord('t'):
            drone.rotate_pitch('down')
        elif key == ord('g'):
            drone.rotate_yaw('left')
        elif key == ord('f'):
            drone.rotate_yaw('right')

cv2.destroyAllWindows()
