import numpy as np
import cv2
from sim_utils.classes import Car, Display, Driver, Drone
import time

if __name__ == '__main__':
    field = cv2.imread('images/grass.jpg')
    field_size = field.shape[:2]

    car1_im = cv2.imread('images/car1s.png')
    car2_im = cv2.imread('images/car2s.png')
    car3_im = cv2.imread('images/car3s.png')

    cars = [Car() for i in range(3)]

    drivers = [Driver(car, field_size) for car in cars]
    cars[0].set_position((1000, 1000))

    display = Display(field, [car1_im, car2_im, car3_im], cars)

    hor_angle = 1.13
    screen_ratio = 0.6
    pitch = 0
    yaw = 0
    altitude = 100
    cam_settings = hor_angle, screen_ratio
    drone = Drone(cam_settings, altitude, pitch, yaw)
    drone.init_position(cars[0].get_position)

    dt = 0.02

    while True:
        for driver in drivers:
            driver.drive(dt)

        display.update_field()
        drone_input = drone.display_visible(display.field_with_cars)

        b, g, r, _ = cv2.mean(drone_input)

        non_green = drone_input.copy().astype(float)
        non_green[:, :, 0] -= b
        non_green[:, :, 1] -= g
        non_green[:, :, 2] -= r
        non_green = np.abs(non_green).astype(np.uint8)

        non_green = np.max(non_green, axis=2)
        gray = np.dstack((non_green, non_green, non_green)).astype(np.uint8)

        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

        cv2.imshow('gray', gray)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)

        corners = cv2.dilate(corners, None)

        drone_input[corners > 0.01 * corners.max()] = [0, 0, 255]

        display.show_field()
        cv2.imshow('corners', drone_input)

        t = drone.transform_view(drone_input)
        cv2.imshow('transfered', t)

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
