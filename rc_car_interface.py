# Copyright(c) Reserved 2020.
# Donghee Lee, University of Soul
#
__author__ = 'will'

import sys
import os
import time

import numpy as np
import cv2

from picamera2 import Picamera2

libcamera_system_path = '/usr/lib/python3/dist-packages'
if libcamera_system_path not in sys.path:
    sys.path.insert(1, libcamera_system_path)


class RC_Car_Interface():

    def __init__(self):
        self.left_wheel = 0
        self.right_wheel = 0
        self.is_moving = False
        self._qr_detector = cv2.QRCodeDetector()
        self.camera = Picamera2()
        
        # Configure camera for still capture with 320x320 resolution
        camera_config = self.camera.create_still_configuration(
            main={"size": (320, 320)}
        )
        self.camera.configure(camera_config)
        self.camera.start()

    def finish_iteration(self):
        print('finish iteration')

    def set_right_speed(self, speed):
        self.right_wheel = speed
        self._update_motion_state()
        print('set right speed to ', speed)
    
    def set_left_speed(self, speed):
        self.left_wheel = speed
        self._update_motion_state()
        print('set left speed to ', speed)

    def _update_motion_state(self):
        self.is_moving = (abs(self.left_wheel) > 0 or abs(self.right_wheel) > 0)
        
    def get_image_from_camera(self):
        # Capture image using picamera2
        img = self.camera.capture_array()
        
        # Convert BGR to grayscale (picamera2 returns RGB, but we need grayscale)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold (same logic as before)
        threshold = int(np.mean(img)) * 0.5

        # Invert black and white with threshold
        ret, img2 = cv2.threshold(img.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY_INV)

        img2 = cv2.resize(img2, (16, 16), interpolation=cv2.INTER_AREA)
        return img2

    def stop(self):     # robot stop
        self.left_wheel = 0
        self.right_wheel = 0
        self.is_moving = False
        print('stop')

    def check_and_stop_on_qr(self):
        """
        Capture a fresh frame, detect QR code, and stop the car if one is found while moving.
        
        Returns:
            (detected: bool, data: str): True와 QR 데이터 문자열을 반환. 미검출 시 (False, "").
        """
        frame = self.camera.capture_array()
        data, points, _ = self._qr_detector.detectAndDecode(frame)

        if data and self.is_moving:
            print(f'QR detected ({data}). Stopping car for 4 seconds.')
            self.stop()
            time.sleep(4)
            print('Resume control after QR stop window.')
            return True, data

        return bool(data), data or ""
        
    def get_raw_image(self):
        """
        원본 320x320 그레이스케일 이미지를 반환합니다 (전처리 없이).
        CNN 모델 훈련 및 추론에 사용됩니다.
        
        Returns:
            numpy array: 320x320 그레이스케일 이미지
        """
        img = self.camera.capture_array()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img
        
    def close(self):
        """Close camera (for cleanup)"""
        if hasattr(self, 'camera') and self.camera:
            self.camera.stop()
            self.camera.close()

# Test Only
# RC_Car_Interface().get_image_from_camera()