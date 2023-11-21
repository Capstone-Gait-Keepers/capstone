
# PURPOSE: This script is used to collect and save data from the ESP8266.

#INSTRUCTIONS: 
#  1. Connect the ESP8266 to your computer via USB.
#  2. Upload the 'arduino_data_capture_accel.ino' sketch to the ESP8266. You should see data coming in the serial monitor in arduino IDE.
#  3. Modify the 'serial_port' variaxble below to match the serial port of your ESP8266. You can find the serial port by looking at the bottom right corner of the arduino IDE. It will be something like '/dev/cu.usbserial-0001' or 'COM5'.
#  4. Run this script to collect data. You will be prompted to enter the number of seconds to collect data and verify that the environment variables are correct. Change them in the code if they are not.

# NOTE: You can stop the data collection early by pressing Ctrl+C.

import serial
import time
from json import dumps
from datetime import datetime
from step_detection import find_steps

from data_types import Recording, RecordingEnvironment, Event, WalkPath

# Name of your serial port (e.g., '/dev/cu.usbserial-0001', or COM5).
def collect_data(seconds: float = None, serial_port = '/dev/cu.usbserial-0001', fs=None, error_threshold=0.5):
    # Open the serial port for communication with the ESP8266.
    ser = serial.Serial(serial_port, 115200)
    t_end = time.time() + seconds if seconds else None # Data collection will run for this many seconds
    vibes = [] # Raw accelerometer data
    events = [] # Timestamps of button presses
    start_time = None
    last_time = None
    expected_period = 1 / fs

    try:
        while not t_end or time.time() < t_end:
            try:
                timestamp, accel, event = ser.readline().decode().strip().split(',')
            except ValueError as e:
                if last_time is not None: # Ignore the first sample being bad
                    raise ValueError('Unable to decode: ' + ser.readline().decode()) from e
            else:
                timestamp = int(timestamp) / 1000
                if last_time is not None and fs is not None:
                    fs_error = (timestamp - last_time) - expected_period
                    if fs_error >= expected_period * error_threshold:
                        raise EnvironmentError(f'Expected {expected_period} sampling period, received {timestamp - last_time}')
                last_time = timestamp
                vibes.append(float(accel))
                if start_time is None:
                    start_time = timestamp
                if event == "BUTTON":
                    events.append(Event('step', timestamp - start_time))
    except KeyboardInterrupt:
        # Close the serial port and the file when you interrupt the script.
        ser.close()
    return vibes, events


if __name__ == "__main__":
    # close_path = WalkPath(start=1.77, stop=2.16, length=3.65)
    # middle_path = WalkPath(start=2.69, stop=3.07, length=3.72)
    # far_path = WalkPath(start=4.28, stop=4.49, length=3.65)

    close_path = WalkPath(start=1.72, stop=1.95, length=3.72)
    middle_path = WalkPath(start=2.74, stop=2.85, length=3.65)
    # ! Change as needed
    env = RecordingEnvironment(
        location='Aarons Studio',
        fs=100,
        floor='cork',
        # obstacle_radius=0.04,
        obstacle_radius=1.38,
        # wall_radius=0.04,
        wall_radius=1.89,
        walk_type='normal',
        # walk_type='shuffle',
        # walk_type='limp',
        # walk_speed='normal',
        walk_speed='slow',
        user='sarah (julia)',
        footwear='socks',
        # footwear='slippers',
        path=close_path,
        # path=middle_path,
    )

    # seconds = input('Input the number of seconds to collect data: ')
    seconds = 100
    # assert input(f'Is this environment correct? \n{dumps(env.to_dict(), sort_keys=True, indent=2)}\n (y/n) ') == 'y', 'Please update the environment in data_capture.py'
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vibes, events = collect_data(int(seconds), fs=env.fs)

    rec = Recording(env, events, vibes)
    # rec.to_yaml(f'datasets/{timestamp}.yaml')
    print(len(find_steps(rec, plot=True)))
