
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

from data_types import Recording, RecordingEnvironment, Event

# Name of your serial port (e.g., '/dev/cu.usbserial-0001', or COM5).
def collect_data(seconds: float, serial_port = '/dev/cu.usbserial-0001', fs=None, error_threshold=0.5):
    # Open the serial port for communication with the ESP8266.
    ser = serial.Serial(serial_port, 115200)
    t_end = time.time() + seconds # Data collection will run for this many seconds
    vibes = [] # Raw accelerometer data
    button_presses = [] # Timestamps of button presses
    start_time = None
    last_time = None
    expected_period = 1/fs

    try:
        while time.time() < t_end:
            try:
                timestamp, accel, event = ser.readline().decode().strip().split(',') # Read data from the serial port.
            except ValueError:
                raise ValueError('Unable to decode: ', ser.readline().decode().strip()) 
            timestamp = int(timestamp)/1000
            if last_time is not None and fs is not None:
                fs_error = (timestamp - last_time) - expected_period
                if fs_error >= expected_period*error_threshold:
                    raise EnvironmentError(f'Expected {expected_period} sampling period, received {timestamp - last_time}')
            last_time = timestamp
            vibes.append(float(accel))
            if start_time is None:
                start_time = timestamp
            if event == "BUTTON":
                button_presses.append(timestamp - start_time)
    except KeyboardInterrupt:
        # Close the serial port and the file when you interrupt the script.
        ser.close()
    return vibes, button_presses


if __name__ == "__main__":
    # ! Change as needed
    env = RecordingEnvironment(
        location='Sunview Living Room',
        fs=100,
        floor='tile',
        obstacle_radius=0.5,
        wall_radius=0.5,
        user='ron',
        footwear='socks',
        temperature=22,
        notes='Testing',
    )

    seconds = input('Input the number of seconds to collect data: ')
    assert input(f'Is this environment correct? \n{dumps(env.to_dict(), sort_keys=True, indent=2)}\n (y/n) ') == 'y', 'Please update the environment in data_capture.py'
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    vibes, button_presses = collect_data(int(seconds), fs=env.fs)

    events = [Event('step', stamp) for stamp in button_presses]
    rec = Recording(env, events, vibes)
    rec.to_yaml(f'datasets/{timestamp}.yaml')
