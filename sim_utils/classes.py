import numpy as np
import cv2
import math
from math import cos, sin
import random
import imutils
import time
import threading

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
        print('setting car position')
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
            curr_position[0] = self.position[0] - 10
        elif dir == 'down':
            curr_position[0] = self.position[0] + 10
        elif dir == 'right':
            curr_position[1] = self.position[1] + 10
        elif dir == 'left':
            curr_position[1] = self.position[1] - 10
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

        points_rotated[:, 0] += self.position[1]
        points_rotated[:, 1] += self.position[0]

        out_width = 300
        out_height = int(out_width * self.screen_ratio)
        # corresponding coordinates of the output image
        out_points = np.float32([[out_width, 0], [0, 0], [out_width, out_height], [0, out_height]])

        m = cv2.getPerspectiveTransform(np.float32(points_rotated), out_points)
        visible = cv2.warpPerspective(field_image, m, (out_width, out_height))

        return visible


class ImageProcessor:
    def __init__(self, cam_settings, pixel_density):
        self.hor_angle, self.screen_ratio = cam_settings
        self.ver_angle = 2 * math.atan(self.screen_ratio * math.tan(self.hor_angle / 2))
        self.altitude = 0
        self.position = 0
        self.pitch = 0
        self.yaw = 0
        self.pixel_density = pixel_density
        self.blob_detector = self.init_blob_detector()

    def init_blob_detector(self):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255

        ################## CHANGE HERE
        # change this to be dependent on pixel density
        params.filterByArea = True
        params.minArea = 400
        params.maxArea = 1000

        params.filterByCircularity = False
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByColor = False

        return cv2.SimpleBlobDetector_create(params)

    def update_altitude(self, altitude):
        self.altitude = altitude

    def update_position(self, position):
        self.position = position

    def update_pitch(self, pitch):
        self.pitch = pitch

    def update_yaw(self, yaw):
        self.yaw = yaw

    def update_drone_readings(self, position, altitude, pitch, yaw):
        self.update_position(position)
        self.update_altitude(altitude)
        self.update_pitch(pitch)
        self.update_yaw(yaw)

    def transform_view(self, image, max_length=1000):
        # max_length is the objects farthest away from drone in 3d cartesian

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
        out_width = int(top_right * 2 * self.pixel_density)
        out_height = int((top_vert_dist - bottom_vert_dist) * self.pixel_density)

        bottom_start_pixel = out_width / 2 - bottom_right * self.pixel_density
        # print(bottom_start_pixel)
        out_points = np.float32([[0, 0], [out_width, 0], [bottom_start_pixel, out_height],
                                 [out_width - bottom_start_pixel, out_height]])
        in_points = np.float32([[0, top_pix_value], [width, top_pix_value], [0, height], [width, height]])

        m = cv2.getPerspectiveTransform(in_points, out_points)
        out = cv2.warpPerspective(image, m, (out_width, out_height))

        return out

    def convert_to_gray(self, image):

        b, g, r, _ = cv2.mean(image)

        # convert image to float and subtract mean color
        non_green = image.copy().astype(float)
        non_green[:, :, 0] -= b
        non_green[:, :, 1] -= g
        non_green[:, :, 2] -= r

        # for each pixel find a channel that maximizes absolute difference from mean
        non_green = np.abs(non_green).astype(np.uint8)
        non_green = np.max(non_green, axis=2)

        # make gray-scale from this image
        gray = np.dstack((non_green, non_green, non_green)).astype(np.uint8)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

        return gray

    def find_cars(self, flat_image, corner_min=1e-5):
        gray = self.convert_to_gray(flat_image)

        # find the corners of the image.
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        mask = (corners > 0.01 * corners.max()).astype(np.uint8) * 255
        mask = np.dstack((mask, mask, mask))

        ################## CHANGE HERE
        # change this to be dependent on pixel density
        kernel = np.ones((30, 30))
        mask = cv2.dilate(mask, kernel)
        mask = cv2.erode(mask, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Setup SimpleBlobDetector parameters.

        keypoints = self.blob_detector.detect(mask)

        contours = []
        for k in keypoints:
            contours.append([int(k.pt[0] - k.size/2), int(k.pt[1] - k.size/2), int(k.size), int(k.size)])

        for c in contours:
            cv2.rectangle(mask, (c[0], c[1]), (c[0] + c[2], c[1] + c[3]), (0, 200, 244), 2)

        cv2.imshow('m', mask)

        return contours

        # mask = corners > 0.01 * corners.max()
        # flat_image[mask] = [0, 255, 255]
        # ch = mask.astype(np.uint8) * 255
        #
        # mask_out = np.dstack((ch, ch, ch))
        # cv2.imshow('mask', mask_out)
        # # cv2.waitKey(0)
        # cv2.imwrite('/home/karol/PycharmProjects/droniada_2019/mask.png', mask_out)
        # # return 0
        # width = 30
        # keypoints = self.find_rectangle(mask.astype(int), width, 10)
        # if keypoints:
        #     x, y = keypoints
        #
        #     return list([min(x), min(y), max(x) - min(x), max(y) - min(y)])

        # for x0, y0 in zip(y, x):
        #     cv2.rectangle(flat_image, (x0, y0), (x0 + width, y0 + width), (155, 155, 0), 2)
        #
        # flat_image[mask] = [0, 155, 155]
        # return flat_image

    def find_rectangle(self, bin_image, width, threshold):
        gray_mask = (bin_image * 255).astype(np.uint8)
        gray = np.dstack((gray_mask, gray_mask, gray_mask))
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        cv2.imshow('g', gray)

        h, w = bin_image.shape
        corners = np.nonzero(bin_image)
        xedges = list(range(0, w, width))
        yedges = list(range(0, h, width))
        hist, _, _ = np.histogram2d(corners[0], corners[1], bins=[yedges, xedges])
        ch = hist.astype(np.uint8)
        hist = np.dstack((ch, ch, ch))
        hist = cv2.resize(hist, (hist.shape[1] * width, hist.shape[0] * width))
        ret, max = cv2.threshold(hist, 20, 255, cv2.THRESH_BINARY)
        # max = np.array(np.where(hist > threshold * hist.mean()), dtype=np.uint8)
        print(max)
        cv2.imshow('hist', max)
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(max)
        print('yo', keypoints)

        return keypoints

    def find_cars_haar(self, flat_image, haar_classifier):
        gray = self.convert_to_gray(flat_image)
        # gray = cv2.cvtColor(flat_image, cv2.COLOR_RGB2GRAY)
        cars = haar_classifier.detectMultiScale(gray, 1.3, 5)

        return cars


class DroneParametersReader(threading.Thread):
    def __init__(self, drone, processor):
        threading.Thread.__init__(self)
        self.drone = drone
        self.processor = processor

    def run(self):
        # this thread will end when the main thread ends
        while True:
            self.processor.update_drone_readings(self.drone.position, self.drone.altitude, self.drone.pitch,
                                                 self.drone.yaw)
            time.sleep(1)
