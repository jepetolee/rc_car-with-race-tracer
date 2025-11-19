# Copyright(c) Reserved 2020.
# Donghee Lee, University of Soul
#
__author__ = 'will'

import numpy as np
import cv2

from picamera2 import Picamera2

class RC_Car_Interface():

    def __init__(self):
        self.left_wheel = 0
        self.right_wheel = 0
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
        print('set right speed to ', speed)
    
    def set_left_speed(self, speed):
        print('set left speed to ', speed)
        
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
        print('stop')
        
    def close(self):
        """Close camera (for cleanup)"""
        if hasattr(self, 'camera') and self.camera:
            self.camera.stop()
            self.camera.close()

# Test Only
# RC_Car_Interface().get_image_from_camera()