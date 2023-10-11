
# INSTRUCTIONS: This script is used to collect data from the ESP8266. It will ask you for a file name and the number of seconds to collect data. It will then collect data for that many seconds and save it to a file with the given name. The data will be saved in the same directory as this script.

# NOTE: You can stop the data collection early by pressing Ctrl+C.


import serial
import time

# Name of your serial port (e.g., '/dev/cu.usbserial-0001', or COM5).
serial_port = '/dev/cu.usbserial-0001'

# Open the serial port for communication with the ESP8266.
ser = serial.Serial(serial_port, 115200)

def collect_data(filename, seconds):
    t_end = time.time() + seconds #data collection will run for this many seconds
    # Open a file to save the data (create it if it doesn't exist).
    with open(filename, 'a') as file:
        try:
            while time.time() < t_end:
                data = ser.readline().decode().strip()  # Read data from the serial port.
                print(data)  # Print the data to the console.
                file.write(data + '\n')  # Write the data to the file.
        except KeyboardInterrupt:
            # Close the serial port and the file when you interrupt the script.
            ser.close()
            file.close()

while True:
    filename = input('Input the file name (description of the data): ') + '.csv'
    seconds = input('Input the number of seconds to collect data: ')
    collect_data(filename, int(seconds))