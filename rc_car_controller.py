#!/usr/bin/env python3
"""
RC Car Controller via Serial Communication
Python script to control RC car via serial communication with Arduino

Command format (Legacy):
- F[speed]: Forward/Gas (e.g., F255)
- L[speed]: Left + Gas (e.g., L150)
- R[speed]: Right + Gas (e.g., R180)
- S: Stop/Coast
- X: Brake

Discrete Action format (CarRacing-v3 Compatible):
- A0: Stop/Coast (do nothing)
- A1: Right + Gas (steer right while accelerating)
- A2: Left + Gas (steer left while accelerating)
- A3: Gas only (straight forward)
- A4: Brake (full stop)
"""

import serial
import time
import sys


# 이산 액션 상수 정의 (CarRacing-v3 호환)
ACTION_STOP = 0       # 정지/코스팅
ACTION_RIGHT_GAS = 1  # 우회전 + 가스
ACTION_LEFT_GAS = 2   # 좌회전 + 가스
ACTION_GAS = 3        # 직진 가스
ACTION_BRAKE = 4      # 브레이크 (RC Car에서는 정지와 동일)

ACTION_NAMES = {
    0: "Stop (Coast)",
    1: "Right + Gas",
    2: "Left + Gas",
    3: "Gas (Forward)",
    4: "Stop (Brake)"  # RC Car에서는 0과 동일하게 처리
}


class RCCarController:
    def __init__(self, port='/dev/ttyACM0', baudrate=9600, delay=0.0):
        """
        Initialize RC car controller
        
        Args:
            port: Serial port (Linux: /dev/ttyUSB0, /dev/ttyACM0, Windows: COM3)
            baudrate: Baud rate (default: 9600)
            delay: Delay after each command in seconds (default: 0.0)
        """
        self.command_delay = delay  # 명령어 간 지연 시간 저장
        try:
            self.serial = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            print(f"Connected to {port} at {baudrate} baud")
            if delay > 0:
                print(f"Command delay: {delay:.3f}s")
            
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
    
    def send_command(self, command, custom_delay=None):
        """
        Send command to Arduino
        
        Args:
            command: Command string (e.g., "F255", "S")
            custom_delay: Optional custom delay (overrides default delay)
        """
        try:
            self.serial.write(f"{command}\n".encode('utf-8'))
            time.sleep(0.05)  # 최소 대기 시간 (Arduino 처리 시간)
            
            # Read response from Arduino
            while self.serial.in_waiting:
                response = self.serial.readline().decode('utf-8').strip()
                if response:
                    print(f"Arduino: {response}")
            
            # 사용자 지정 지연 시간 적용
            delay_time = custom_delay if custom_delay is not None else self.command_delay
            if delay_time > 0:
                time.sleep(delay_time)
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def execute_discrete_action(self, action):
        """
        이산 액션 실행 (CarRacing-v3 호환)
        
        Args:
            action: 이산 액션 (0-4)
                0: Stop/Coast (정지) → 정지
                1: Right + Gas (우회전 + 가스)
                2: Left + Gas (좌회전 + 가스)
                3: Gas (직진 가스)
                4: Brake (브레이크) → 정지 (0과 동일)
        
        Note:
            CarRacing에서 학습 시 action 0과 4는 다르게 취급되지만,
            RC Car 실행 시에는 둘 다 정지로 매핑됨
        """
        action = int(action)
        if action < 0 or action > 4:
            print(f"Invalid action: {action}")
            return
        
        action_name = ACTION_NAMES.get(action, "Unknown")
        print(f"Action {action}: {action_name}")
        self.send_command(f"A{action}")
    
    def forward(self, speed=200):
        """Move forward / Gas (speed: 0-255)"""
        speed = max(0, min(255, speed))
        print(f"Forward - Speed: {speed}")
        self.send_command(f"F{speed}")
    
    def gas(self, speed=200):
        """Alias for forward - straight gas"""
        self.forward(speed)
    
    def left_gas(self, speed=200):
        """Turn left while accelerating (speed: 0-255)"""
        speed = max(0, min(255, speed))
        print(f"Left + Gas - Speed: {speed}")
        self.send_command(f"L{speed}")
    
    def right_gas(self, speed=200):
        """Turn right while accelerating (speed: 0-255)"""
        speed = max(0, min(255, speed))
        print(f"Right + Gas - Speed: {speed}")
        self.send_command(f"R{speed}")
    
    # Legacy aliases
    def left(self, speed=200):
        """Turn left (alias for left_gas)"""
        self.left_gas(speed)
    
    def right(self, speed=200):
        """Turn right (alias for right_gas)"""
        self.right_gas(speed)
    
    def stop(self):
        """Stop / Coast (RC Car에서 brake와 동일)"""
        print("Stop")
        self.send_command("S")
    
    def brake(self):
        """Brake (RC Car에서 stop과 동일하게 처리)"""
        print("Brake → Stop")
        self.send_command("S")  # RC Car에서는 stop과 동일
    
    def close(self):
        """Close serial port"""
        if self.serial.is_open:
            self.stop()  # Stop before closing
            self.serial.close()
            print("Connection closed")


def interactive_mode(controller):
    """Interactive mode - control with keyboard (CarRacing compatible)"""
    print("\n=== RC Car Interactive Control Mode ===")
    print("=== CarRacing-v3 Compatible (No Reverse) ===")
    print("Commands:")
    print("  w: Forward (Gas)")
    print("  a: Left + Gas")
    print("  d: Right + Gas")
    print("  s: Stop/Coast")
    print("  x: Brake")
    print("  0-4: Discrete actions")
    print("     0: Stop, 1: Right+Gas, 2: Left+Gas, 3: Gas, 4: Brake")
    print("  q: Quit")
    print("==========================================\n")
    
    speed = 200  # Default speed
    
    try:
        while True:
            cmd = input("Enter command (w/a/d/s/x/0-4/q): ").strip().lower()
            
            if cmd == 'w':
                controller.forward(speed)
            elif cmd == 'a':
                controller.left_gas(speed)
            elif cmd == 'd':
                controller.right_gas(speed)
            elif cmd == 's':
                controller.stop()
            elif cmd == 'x':
                controller.brake()
            elif cmd in ['0', '1', '2', '3', '4']:
                controller.execute_discrete_action(int(cmd))
            elif cmd == 'q':
                print("Exiting...")
                break
            elif cmd.startswith('speed '):
                try:
                    new_speed = int(cmd.split()[1])
                    speed = max(0, min(255, new_speed))
                    print(f"Speed changed: {speed}")
                except:
                    print("Invalid speed value.")
            else:
                print("Unknown command. Use w/a/d/s/x or 0-4")
    except KeyboardInterrupt:
        print("\nExiting...")


def demo_mode(controller, step_delay=1.0):
    """
    Demo mode - automatic driving test (CarRacing style)
    
    Args:
        controller: RCCarController instance
        step_delay: Delay between demo steps in seconds (default: 1.0)
    """
    print("\n=== RC Car Demo Mode (CarRacing Style) ===")
    
    print("1. Gas/Forward (2 seconds)")
    controller.execute_discrete_action(ACTION_GAS)  # Action 3
    time.sleep(2)
    
    print("2. Right + Gas (1.5 seconds)")
    controller.execute_discrete_action(ACTION_RIGHT_GAS)  # Action 1
    time.sleep(1.5)
    
    print("3. Gas/Forward (2 seconds)")
    controller.execute_discrete_action(ACTION_GAS)  # Action 3
    time.sleep(2)
    
    print("4. Left + Gas (1.5 seconds)")
    controller.execute_discrete_action(ACTION_LEFT_GAS)  # Action 2
    time.sleep(1.5)
    
    print("5. Gas/Forward (2 seconds)")
    controller.execute_discrete_action(ACTION_GAS)  # Action 3
    time.sleep(2)
    
    print("6. Coast/Stop (1 second)")
    controller.execute_discrete_action(ACTION_STOP)  # Action 0
    time.sleep(1)
    
    print("7. Brake")
    controller.execute_discrete_action(ACTION_BRAKE)  # Action 4
    
    print("\nDemo completed!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RC Car Controller')
    parser.add_argument('--port', default='/dev/ttyACM0', 
                        help='Serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--baudrate', type=int, default=9600,
                        help='Baud rate (default: 9600)')
    parser.add_argument('--mode', choices=['interactive', 'demo'], 
                        default='interactive',
                        help='Execution mode: interactive (keyboard control) or demo (automatic test)')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='Delay after each command in seconds (default: 0.0, recommended: 0.1 for smooth control)')
    
    args = parser.parse_args()
    
    # Initialize RC Car controller
    controller = RCCarController(port=args.port, baudrate=args.baudrate, delay=args.delay)
    
    try:
        if args.mode == 'interactive':
            interactive_mode(controller)
        elif args.mode == 'demo':
            demo_mode(controller, step_delay=args.delay if args.delay > 0 else 1.0)
    finally:
        controller.close()


if __name__ == "__main__":
    main()

