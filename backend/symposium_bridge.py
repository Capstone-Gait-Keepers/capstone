import serial
import time
import json
import sys
import os
from data_analysis.data_types import Recording, get_optimal_analysis_params, SensorType
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_analysis'))
from data_analysis.metric_analysis import AnalysisController

START_SIGNAL = b'BEGIN_TRANSMISSION'
ctrl = AnalysisController(**get_optimal_analysis_params(SensorType.PIEZO))


def serial_bridge(serial_port='/dev/cu.usbserial-0001', baud_rate=115200):
    # Open the serial port for communication with the ESP8266.
    ser = serial.Serial(serial_port, baud_rate)
    line = ser.readline()
    while True:
        try:
            while START_SIGNAL not in line:
                line = ser.readline()
                if line:
                    print(line)
                # TODO: Need sleep to reduce CPU usage?
                time.sleep(0.05)
            line = b''
            url = ser.readline().decode().strip()
            data_test = ser.readline().strip()
            json_ts_data = json.loads(data_test)['ts_data']
            print(f"Data length {len(json_ts_data)}")

            # plot the data
            rec = Recording.from_real_data(fs=500, data=json_ts_data)
            # Recording.from_real_data(fs=500, data=json_ts_data).plot()
            metrics = ctrl.get_recording_metrics(rec, plot=True)[0]
            print(metrics)
            
            
            # TODO: Validate JSON and url
            # resp = requests.post(url, json=json_ts_data)
            # print(f"{resp.status_code}: {resp.text}")

        # TODO: Confirm error handling works
        except UnicodeDecodeError:
            print(f"500: Couldn't decode")



if __name__ == '__main__':
    serial_bridge()
