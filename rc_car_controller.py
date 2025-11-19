#!/usr/bin/env python3
"""
RC Car Controller via Serial Communication
Python script to control RC car via serial communication with Arduino

Command format:
- F[speed]: Forward (e.g., F255)
- B[speed]: Backward (e.g., B200)
- L[speed]: Turn left (e.g., L150)
- R[speed]: Turn right (e.g., R180)
- S: Stop
"""

import serial
import time
import sys


class RCCarController:
    def __init__(self, port='/dev/ttyUSB0', baudrate=9600):
        """
        Initialize RC car controller
        
        Args:
            port: Serial port (Linux: /dev/ttyUSB0, /dev/ttyACM0, Windows: COM3)
            baudrate: Baud rate (default: 9600)
        """
        try:
            self.serial = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            print(f"Connected to {port} at {baudrate} baud")
            
            # Read initial message from Arduino
            if self.serial.in_waiting:
                response = self.serial.readline().decode('utf-8').strip()
                print(f"Arduino: {response}")
        except serial.SerialException as e:
            print(f"Error: Cannot open serial port - {e}")
            print("Please check available ports:")
            print("  Linux: /dev/ttyUSB0, /dev/ttyACM0")
            print("  Windows: COM3, COM4, etc.")
            sys.exit(1)
    
    def send_command(self, command):
        """
        Send command to Arduino
        
        Args:
            command: Command string (e.g., "F255", "S")
        """
        try:
            self.serial.write(f"{command}\n".encode('utf-8'))
            time.sleep(0.1)  # Wait for command processing
            
            # Read response from Arduino
            while self.serial.in_waiting:
                response = self.serial.readline().decode('utf-8').strip()
                if response:
                    print(f"Arduino: {response}")
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def forward(self, speed=200):
        """Move forward (speed: 0-255)"""
        speed = max(0, min(255, speed))  # Limit speed range
        print(f"Forward - Speed: {speed}")
        self.send_command(f"F{speed}")
    
    def backward(self, speed=200):
        """Move backward (speed: 0-255)"""
        speed = max(0, min(255, speed))
        print(f"Backward - Speed: {speed}")
        self.send_command(f"B{speed}")
    
    def left(self, speed=200):
        """Turn left (speed: 0-255)"""
        speed = max(0, min(255, speed))
        print(f"Turn left - Speed: {speed}")
        self.send_command(f"L{speed}")
    
    def right(self, speed=200):
        """Turn right (speed: 0-255)"""
        speed = max(0, min(255, speed))
        print(f"Turn right - Speed: {speed}")
        self.send_command(f"R{speed}")
    
    def stop(self):
        """Stop"""
        print("Stop")
        self.send_command("S")
    
    def close(self):
        """Close serial port"""
        if self.serial.is_open:
            self.stop()  # Stop before closing
            self.serial.close()
            print("Connection closed")


def interactive_mode(controller):
    """Interactive mode - control with keyboard"""
    print("\n=== RC Car Interactive Control Mode ===")
    print("Commands:")
    print("  w: Forward")
    print("  s: Backward")
    print("  a: Turn left")
    print("  d: Turn right")
    print("  x: Stop")
    print("  q: Quit")
    print("===================================\n")
    
    speed = 200  # Default speed
    
    try:
        while True:
            cmd = input("Enter command (w/s/a/d/x/q): ").strip().lower()
            
            if cmd == 'w':
                controller.forward(speed)
            elif cmd == 's':
                controller.backward(speed)
            elif cmd == 'a':
                controller.left(speed)
            elif cmd == 'd':
                controller.right(speed)
            elif cmd == 'x':
                controller.stop()
            elif cmd == 'q':
                print("Exiting...")
                break
            elif cmd.startswith('speed '):
                # Change speed (e.g., speed 150)
                try:
                    new_speed = int(cmd.split()[1])
                    speed = max(0, min(255, new_speed))
                    print(f"Speed changed: {speed}")
                except:
                    print("Invalid speed value.")
            else:
                print("Unknown command.")
    except KeyboardInterrupt:
        print("\nExiting...")


def demo_mode(controller):
    """Demo mode - automatic driving test"""
    print("\n=== RC Car Demo Mode ===")
    
    print("1. Forward (3 seconds)")
    controller.forward(200)
    time.sleep(3)
    
    print("2. Stop (1 second)")
    controller.stop()
    time.sleep(1)
    
    print("3. Backward (2 seconds)")
    controller.backward(150)
    time.sleep(2)
    
    print("4. Stop (1 second)")
    controller.stop()
    time.sleep(1)
    
    print("5. Turn left (2 seconds)")
    controller.left(180)
    time.sleep(2)
    
    print("6. Stop (1 second)")
    controller.stop()
    time.sleep(1)
    
    print("7. Turn right (2 seconds)")
    controller.right(180)
    time.sleep(2)
    
    print("8. Stop")
    controller.stop()
    
    print("\nDemo completed!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RC Car Controller')
    parser.add_argument('--port', default='/dev/ttyUSB0', 
                        help='Serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--baudrate', type=int, default=9600,
                        help='Baud rate (default: 9600)')
    parser.add_argument('--mode', choices=['interactive', 'demo'], 
                        default='interactive',
                        help='Execution mode: interactive (keyboard control) or demo (automatic test)')
    
    args = parser.parse_args()
    
    # Initialize RC Car controller
    controller = RCCarController(port=args.port, baudrate=args.baudrate)
    
    try:
        if args.mode == 'interactive':
            interactive_mode(controller)
        elif args.mode == 'demo':
            demo_mode(controller)
    finally:
        controller.close()


if __name__ == "__main__":
    main()

