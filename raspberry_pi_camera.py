#!/usr/bin/env python3
"""
Raspberry Pi Camera Test Script
Standalone script to test and display Raspberry Pi camera feed

Features:
- Real-time video preview
- Image capture
- Configurable resolution and frame rate
- Black and white mode support
"""

import numpy as np
import cv2
import time
import argparse
import sys

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False
    print("Warning: picamera2 module not found. This script requires Raspberry Pi with picamera2 module.")
    print("Install with: sudo apt-get install python3-picamera2")
    sys.exit(1)


class RaspberryPiCamera:
    def __init__(self, resolution=(320, 320), framerate=30, grayscale=False):
        """
        Initialize Raspberry Pi Camera
        
        Args:
            resolution: Camera resolution (width, height)
            framerate: Frame rate (fps)
            grayscale: If True, convert images to grayscale during processing
        """
        self.camera = Picamera2()
        self.resolution = resolution
        self.framerate = framerate
        self.grayscale = grayscale
        
        # Configure camera for video preview
        video_config = self.camera.create_video_configuration(
            main={"size": resolution, "format": "RGB888"}
        )
        self.camera.configure(video_config)
        self.camera.start()
        
        # Warm up camera
        time.sleep(2)
        print(f"Camera initialized: {resolution[0]}x{resolution[1]} @ {framerate}fps")
        if grayscale:
            print("Grayscale mode enabled (processing)")
    
    def capture_image(self, save_path=None):
        """
        Capture a single image from camera
        
        Args:
            save_path: Optional path to save image (if None, returns image array)
        
        Returns:
            numpy array of image (RGB format)
        """
        # Capture array from camera
        img = self.camera.capture_array()
        
        # Convert RGB to BGR for OpenCV compatibility
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if save_path:
            cv2.imwrite(save_path, img_bgr)
            print(f"Image saved to: {save_path}")
        
        return img_bgr
    
    def get_image_grayscale(self):
        """
        Get grayscale image (16x16 processed, similar to rc_car_interface)
        
        Returns:
            16x16 grayscale image array
        """
        # Capture image from camera
        img = self.camera.capture_array()
        
        # Convert RGB to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        threshold = int(np.mean(img)) * 0.5
        ret, img2 = cv2.threshold(img.astype(np.uint8), threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Resize to 16x16
        img2 = cv2.resize(img2, (16, 16), interpolation=cv2.INTER_AREA)
        
        return img2
    
    def video_preview(self, duration=None, show_processed=False):
        """
        Display real-time video preview
        
        Args:
            duration: Preview duration in seconds (None for infinite)
            show_processed: If True, also show processed 16x16 image
        """
        start_time = time.time()
        print("Starting video preview. Press 'q' to quit, 's' to save snapshot")
        
        try:
            for frame in self.camera.frames():
                # Get frame array (RGB format)
                image = frame.array
                
                # Convert RGB to BGR for OpenCV display
                display_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Convert to grayscale if requested
                if self.grayscale:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                cv2.imshow("Raspberry Pi Camera - Original", display_img)
                
                # Display processed image if requested
                if show_processed:
                    # Get processed image (this will capture a new frame)
                    processed = self.get_image_grayscale()
                    processed_display = cv2.resize(processed, (320, 320), interpolation=cv2.INTER_NEAREST)
                    processed_display = cv2.cvtColor(processed_display, cv2.COLOR_GRAY2BGR)
                    cv2.imshow("Raspberry Pi Camera - Processed (16x16)", processed_display)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"camera_snapshot_{timestamp}.jpg"
                    cv2.imwrite(filename, display_img)
                    print(f"Snapshot saved: {filename}")
                
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                    
        except KeyboardInterrupt:
            print("\nPreview interrupted")
        finally:
            cv2.destroyAllWindows()
    
    def close(self):
        """Close camera"""
        if hasattr(self, 'camera') and self.camera:
            self.camera.stop()
            self.camera.close()
        print("Camera closed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Raspberry Pi Camera Test Script')
    parser.add_argument('--mode', choices=['preview', 'capture', 'test'], 
                        default='preview',
                        help='Mode: preview (video), capture (single image), or test (processed image)')
    parser.add_argument('--resolution', type=str, default='320,320',
                        help='Camera resolution as width,height (default: 320,320)')
    parser.add_argument('--framerate', type=int, default=30,
                        help='Frame rate in fps (default: 30)')
    parser.add_argument('--grayscale', action='store_true',
                        help='Enable grayscale mode')
    parser.add_argument('--duration', type=int, default=None,
                        help='Preview duration in seconds (default: infinite)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path for capture mode')
    parser.add_argument('--show-processed', action='store_true',
                        help='Show processed 16x16 image in preview mode')
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split(','))
        resolution = (width, height)
    except:
        print("Invalid resolution format. Use width,height (e.g., 320,320)")
        sys.exit(1)
    
    # Check if running on Raspberry Pi
    if not HAS_PICAMERA2:
        print("Error: picamera2 module is required. This script must run on Raspberry Pi.")
        sys.exit(1)
    
    # Initialize camera
    camera = RaspberryPiCamera(
        resolution=resolution,
        framerate=args.framerate,
        grayscale=args.grayscale
    )
    
    try:
        if args.mode == 'preview':
            # Video preview mode
            camera.video_preview(duration=args.duration, show_processed=args.show_processed)
        
        elif args.mode == 'capture':
            # Single image capture
            output_path = args.output or f"camera_capture_{int(time.time())}.jpg"
            img = camera.capture_image(save_path=output_path)
            print(f"Captured image shape: {img.shape}")
        
        elif args.mode == 'test':
            # Test processed image (like rc_car_interface)
            print("Capturing and processing image...")
            processed_img = camera.get_image_grayscale()
            print(f"Processed image shape: {processed_img.shape}")
            
            # Display processed image
            display_img = cv2.resize(processed_img, (320, 320), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Processed Image (16x16)", display_img)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Save if output specified
            if args.output:
                cv2.imwrite(args.output, processed_img)
                print(f"Processed image saved to: {args.output}")
    
    finally:
        camera.close()


if __name__ == "__main__":
    main()

