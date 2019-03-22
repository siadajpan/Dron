import numpy as np
import cv2
import math
from math import cos, sin
import random
import imutils

MAX_ANGLE_CHANGE = math.pi/30
MAX_SPEED = 85                  # in dm / s (pixels / s)


class Car:
    def __init__(self):
        self._position = (0, 0)
        self._direction = 0      # direction in rad
        self._speed = 0          # speed in m/s
        self._angle_change = 0
        self._color = 'unknown'

    def set_position(self, position):
        self._position = position

    def set_direction(self, direction):
        self._direction = direction

    def set_speed(self, speed):
        self._speed = speed

    def set_angle_change(self, angle_change):
        self._angle_change = angle_change

    def set_color(self, color):
        self._color = color

    @property
    def get_position(self):
        return self._position

    @property
    def get_direction(self):
        return self._direction

    @property
    def get_angle_change(self):
        return self._angle_change

    def init_drive(self, pos, direction, speed, angle_change):
        self.set_position(pos)
        self.set_direction(direction)
        self.set_speed(speed)
        self.set_angle_change(angle_change)

    def update_position(self, dt):
        # change position based on the time difference from last update

        pos_x, pos_y = self._position

        distance = self._speed * dt

        self.set_direction(self._direction + math.tan(self._angle_change) * distance / 20)

        pos_x += distance * cos(self._direction)
        pos_y += distance * sin(self._direction)
        self.set_position((pos_x, pos_y))


class Driver:
    def __init__(self, car, field_size):
        self.car = car
        self.field_size = field_size
        self.init_car_drive()

    def init_car_drive(self):
        field_size_x, field_size_y = self.field_size
        pos_x = random.random() * field_size_x * 0.6 + 0.2 * field_size_x
        pos_y = random.random() * field_size_y * 0.6 + 0.2 * field_size_x
        self.car.set_position((pos_x, pos_y))
        self.car.set_direction(random.random() * 2 * math.pi)
        self.car.set_speed(random.random() * MAX_SPEED)
        self.car.set_angle_change(abs(random.random()) * MAX_ANGLE_CHANGE/10)

    @property
    def check_border(self):
        # check if car is on the border of the filed
        # return True if car is on the border, false if not
        field_size_x, field_size_y = self.field_size
        border = min(field_size_x, field_size_y) / 3
        pos_x, pos_y = self.car.get_position
        outside_x = pos_x + border >= field_size_x or pos_x < border
        outside_y = pos_y + border >= field_size_y or pos_y < border

        return outside_x or outside_y

    def random_drive(self):
        r = random.random()

        if r > 0.9:
            self.car.set_angle_change((random.random() - 0.5) * MAX_ANGLE_CHANGE)

        if r > 0.95:
            self.car.set_speed(abs(random.random()) * MAX_SPEED)

        if r > 0.99:
            self.car.set_speed(0)

        if abs(self.car.get_angle_change) >= MAX_ANGLE_CHANGE - 0.1:
            self.car.set_angle_change((random.random() - 0.5) * MAX_ANGLE_CHANGE / 2)

    def drive(self, time_delay):
        if not self.check_border:
            self.random_drive()
        # car is on border
        else:
            self.car.set_speed(MAX_SPEED / 2)
            self.car.set_angle_change(-MAX_ANGLE_CHANGE)

        self.car.update_position(time_delay)


class Display:
    def __init__(self, field_im, cars_im, cars_obj):
        self.field = field_im
        self.field_with_cars = field_im.copy()
        self.cars_im = cars_im
        self.cars = cars_obj

    def update_field(self):
        self.field_with_cars = self.field.copy()
        for (car, car_im) in zip(self.cars, self.cars_im):
            image_size = car_im.shape[:2]
            w, h = image_size
            car_rotated = imutils.rotate(car_im, car.get_direction * 180 / math.pi - 90, (int(h/2), int(w/2)))
            alpha = cv2.cvtColor(car_rotated, cv2.COLOR_RGB2GRAY) > 0
            alpha = np.dstack((alpha, alpha, alpha))

            image_position = np.array(car.get_position, dtype=int) - [int(h/2), int(w/2)]
            x, y = image_position
            try:
                overlay = self.field_with_cars[x: x + w, y: y + w]
                self.field_with_cars[x: x + w, y: y + w] = overlay * (1 - alpha) + car_rotated * alpha
            except ValueError:
                continue

    def show_field(self):
        new_h, new_w = (np.array(self.field_with_cars.shape[:2]) / 3).astype(int)
        cv2.imshow('sim', cv2.resize(self.field_with_cars, dsize=(new_w, new_h)))


class Drone:
    def __init__(self, cam_settings, altitude, pitch, yaw):
        self.hor_angle, self.screen_ratio = cam_settings
        self.altitude = altitude
        self.yaw = yaw
        self.pitch = pitch
        self.ver_angle = 2 * math.atan(self.screen_ratio * math.tan(self.hor_angle / 2))
        self.position = (0, 0)

    def init_position(self, position):
        self.position = position

    def increase_altitude(self):
        self.altitude += 50

    def decrease_altitude(self):
        self.altitude -= 50
        self.altitude = max(self.altitude, 10)

    def move(self, dir):
        curr_position = np.array(self.position)
        if dir == 'up':
            curr_position[1] = self.position[1] - 10
        elif dir == 'down':
            curr_position[1] = self.position[1] + 10
        elif dir == 'right':
            curr_position[0] = self.position[0] + 10
        elif dir == 'left':
            curr_position[0] = self.position[0] - 10
        self.position = tuple(curr_position)

    def rotate_pitch(self, dir):
        if dir == 'up':
            self.pitch += 0.1
        elif dir == 'down':
            self.pitch -= 0.1

    def rotate_yaw(self, dir):
        if dir == 'left':
            self.yaw += 0.1
        elif dir == 'right':
            self.yaw -= 0.1

    def display_visible(self, field_image):

        top_angle = self.pitch + self.ver_angle / 2
        bottom_angle = self.pitch - self.ver_angle / 2

        top_angle = np.clip(top_angle, -math.pi / 2, math.pi / 2)
        bottom_angle = np.clip(bottom_angle, -math.pi / 2, math.pi / 2)

        top = self.altitude * math.tan(top_angle)
        bottom = self.altitude * math.tan(bottom_angle)

        dist_cam_top = math.sqrt(self.altitude ** 2 + top ** 2)
        dist_cam_bottom = math.sqrt(self.altitude ** 2 + bottom ** 2)

        top_right = dist_cam_top * math.tan(self.hor_angle / 2)
        bottom_right = dist_cam_bottom * math.tan(self.hor_angle / 2)

        # adding - before top and bottom, as image starts from 0 and going down adding numbers
        points = np.array([[top_right, -top], [-top_right, -top], [bottom_right, -bottom], [-bottom_right, -bottom]])

        rot = cv2.getRotationMatrix2D((0, 0), self.yaw * 180 / math.pi, 1)
        points_rotated = np.dot(points, rot[:2, :2])

        points_rotated[:, 0] += self.position[0]
        points_rotated[:, 1] += self.position[1]

        out_width = 300
        out_height = int(out_width * self.screen_ratio)
        # corresponding coordinates of the output image
        out_points = np.float32([[out_width, 0], [0, 0], [out_width, out_height], [0, out_height]])

        m = cv2.getPerspectiveTransform(np.float32(points_rotated), out_points)
        visible = cv2.warpPerspective(field_image, m, (out_width, out_height))

        return visible

    def transform_view(self, image, max_length=1000, pixel_density=1):
        # max_length is the objects farthest away from drone in 3d cartesian
        # cam_settings = (angle_of_view, screen_ratio, pitch)

        if self.altitude > max_length:
            print('altitude too high')
            return image

        height, width = image.shape[:2]

        bottom_angle = self.pitch - self.ver_angle / 2
        top_angle = self.pitch + self.ver_angle / 2

        # angle that correspond to the farthest object
        max_length_angle = math.acos(self.altitude / max_length)

        if max_length_angle < bottom_angle:
            return image

        bottom_distance = self.altitude / math.cos(bottom_angle)
        bottom_right = bottom_distance * math.tan(self.hor_angle / 2)
        bottom_vert_dist = self.altitude * math.tan(bottom_angle)

        if max_length_angle < top_angle:
            top_pix_value = height/2 * (1 - math.tan(max_length_angle - self.pitch) / math.tan(top_angle - self.pitch))
            top_angle = max_length_angle
        else:
            top_pix_value = 0

        top_distance = self.altitude / math.cos(top_angle)
        top_right = top_distance * math.tan(self.hor_angle / 2)
        top_vert_dist = self.altitude * math.tan(top_angle)

        # end_hor_distance should always be bigger if pitch >= 0
        out_width = int(top_right * 2 * pixel_density)
        out_height = int((top_vert_dist - bottom_vert_dist) * pixel_density)

        bottom_start_pixel = out_width / 2 - bottom_right * pixel_density
        # print(bottom_start_pixel)
        out_points = np.float32([[0, 0], [out_width, 0], [bottom_start_pixel, out_height],
                                 [out_width - bottom_start_pixel, out_height]])
        in_points = np.float32([[0, top_pix_value], [width, top_pix_value], [0, height], [width, height]])

        m = cv2.getPerspectiveTransform(in_points, out_points)
        out = cv2.warpPerspective(image, m, (out_width, out_height))

        return out
