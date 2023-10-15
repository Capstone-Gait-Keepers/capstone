
# PURPOSE: This script is used to collect and save data from the ESP8266.

#INSTRUCTIONS: 
#  1. Connect the ESP8266 to your computer via USB.
#  2. Upload the 'arduino_data_capture_accel.ino' sketch to the ESP8266. You should see data coming in the serial monitor in arduino IDE.
#  3. Modify the 'serial_port' variaxble below to match the serial port of your ESP8266. You can find the serial port by looking at the bottom right corner of the arduino IDE. It will be something like '/dev/cu.usbserial-0001' or 'COM5'.
#  4. Run this script to collect data. You will be prompted to enter a file name and the number of seconds to collect data.

# NOTE: You can stop the data collection early by pressing Ctrl+C.

import serial
import time

# Name of your serial port (e.g., '/dev/cu.usbserial-0001', or COM5).
serial_port = '/dev/cu.usbserial-0001'

def collect_data(filename, seconds):
    # Open the serial port for communication with the ESP8266.
    ser = serial.Serial(serial_port, 115200)
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


filename = 'datasets/' + input('Input the file name (description of the data): ') + '.csv'
seconds = input('Input the number of seconds to collect data: ')
collect_data(filename, int(seconds))