import serial
import time
import json
import sys
import os
import numpy as np
from data_analysis.data_types import Recording, get_optimal_analysis_params, SensorType, RecordingEnvironment, WalkPath
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_analysis'))
from data_analysis.metric_analysis import AnalysisController

START_SIGNAL = b'BEGIN_TRANSMISSION'
ctrl = AnalysisController(**get_optimal_analysis_params(SensorType.PIEZO, fs=500))


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
                time.sleep(0.05)
            line = b''
            url = ser.readline().decode().strip()
            data_test = ser.readline().strip()
            json_ts_data = json.loads(data_test)['ts_data']
            max_amp = np.max(np.abs(json_ts_data))
            if max_amp < 0.1:
                print(f"Max amplitude too low ({max_amp}), skipping")
            else:
                print(f"Data length {len(json_ts_data)}")
                rec = Recording.from_real_data(fs=ctrl.fs, data=json_ts_data)
                ctrl.get_recording_metrics(rec, plot_with_metrics=True)
                if input('Save?') == 'y':
                    index = 2
                    while os.path.exists(f"data-{index}.yml"):
                        index += 1
                    rec.env = RecordingEnvironment(
                        ctrl.fs,
                        location="Custom Floor",
                        user="ron",
                        floor="plywood",
                        footwear="socks",
                        walk_type="normal",
                        obstacle_radius=0,
                        path=WalkPath(
                            start=1.8,
                            stop=1.8,
                            length=3.0,
                        ),
                        wall_radius=0,
                        quality="normal",
                        walk_speed="normal",
                        # notes="Walked off",
                    )
                    rec.to_file(f'data-{index}.yml')
        except UnicodeDecodeError:
            print(f"500: Couldn't decode")



if __name__ == '__main__':
    serial_bridge()
    # rec = Recording.from_file('TEST-0.yml')
    # rec.plot()
