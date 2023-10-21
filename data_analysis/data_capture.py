
# PURPOSE: This script is used to collect and save data from the ESP8266.

#INSTRUCTIONS: 
#  1. Connect the ESP8266 to your computer via USB.
#  2. Upload the 'arduino_data_capture_accel.ino' sketch to the ESP8266. You should see data coming in the serial monitor in arduino IDE.
#  3. Modify the 'serial_port' variaxble below to match the serial port of your ESP8266. You can find the serial port by looking at the bottom right corner of the arduino IDE. It will be something like '/dev/cu.usbserial-0001' or 'COM5'.
#  4. Run this script to collect data. You will be prompted to enter a file name and the number of seconds to collect data.

# NOTE: You can stop the data collection early by pressing Ctrl+C.

import serial
import time
from json import dumps
from datetime import datetime

from data_types import Recording, RecordingEnvironment, Event, Source


# Name of your serial port (e.g., '/dev/cu.usbserial-0001', or COM5).
def collect_data(seconds: float, serial_port = '/dev/cu.usbserial-0001'):
    # Open the serial port for communication with the ESP8266.
    ser = serial.Serial(serial_port, 115200)
    t_end = time.time() + seconds # Data collection will run for this many seconds
    vibes = [] # Raw accelerometer data
    button_presses = [] # Timestamps of button presses
    start_time = None

    # Open a file to save the data (create it if it doesn't exist).
    try:
        while time.time() < t_end:
            # TODO: Data should include timestamps and button presses. Split data by commas?
            data = ser.readline().decode().strip()  # Read data from the serial port.
            print(data) # Print the data to the console.
            vibes.append(float(data))
            # if start_time is None:
            #     start_time = data
            # if data == 'button':
            #   button_presses.append(data - start_time)
    except KeyboardInterrupt:
        # Close the serial port and the file when you interrupt the script.
        ser.close()
    return vibes, button_presses


if __name__ == "__main__":
    # filename = 'datasets/' + input('Input the file name (description of the data): ') + '.csv'
    # collect_data(filename, int(seconds))

    # ! Change as needed
    env = RecordingEnvironment(
        location='Tals bedroom',
        fs=100,
        floor='hardwood',
        obstacle_radius=0.5,
        wall_radius=0.5,
    )
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    seconds = input('Input the number of seconds to collect data: ')
    assert input(f'Is this environment correct? \n{dumps(env.to_dict(), sort_keys=True, indent=2)}\n (y/n) ') == 'y', 'Please update the environment in data_capture.py'
    vibes, button_presses = collect_data(int(seconds))

    events = [Event('step', t, source=Source.USER) for t in button_presses]
    rec = Recording(env, events, vibes)
    rec.to_yaml(f'datasets/{timestamp}.yaml')
