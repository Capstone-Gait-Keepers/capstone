import serial
import requests
import time

START_SIGNAL = "BEGIN_TRANSMISSION"

def serial_bridge(serial_port='COM8', baud_rate=115200):
    # Open the serial port for communication with the ESP8266.
    ser = serial.Serial(serial_port, baud_rate)
    line = read_line(ser)
    while True:
        try:
            while line != START_SIGNAL:
                line = read_line(ser)
                # TODO: Need sleep to reduce CPU usage?
                time.sleep(0.05)
            url = read_line(ser)
            json = read_line(ser)
            # TODO: Validate JSON and url
            resp = requests.post(url, json=json)
            ser.write(f"{resp.status_code}: {resp.text}")

        # TODO: Confirm error handling works
        except UnicodeDecodeError:
            ser.write(f"500: Couldn't decode")
        except requests.exceptions.ConnectionError:
            ser.write(f"400: Couldn't connect to {url}")
        except requests.exceptions.InvalidURL:
            ser.write(f"400: Invalid URL ({url})")


def read_line(ser: serial.Serial):
    return ser.readline().decode().strip()


if __name__ == '__main__':
    serial_bridge()