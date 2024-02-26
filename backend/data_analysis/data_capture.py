
# PURPOSE: This script is used to collect and save data from the ESP8266.

#INSTRUCTIONS: 
#  1. Connect the ESP8266 to your computer via USB.
#  2. Upload the 'arduino_data_capture_accel.ino' sketch to the ESP8266. You should see data coming in the serial monitor in arduino IDE.
#  3. Modify the 'serial_port' variable below to match the serial port of your ESP8266. You can find the serial port by looking at the bottom right corner of the arduino IDE. It will be something like '/dev/cu.usbserial-0001' or 'COM5'.
#  4. Run this script to collect data. You will be prompted to enter the number of seconds to collect data and verify that the environment variables are correct. Change them in the code if they are not.

# NOTE: You can stop the data collection early by pressing Ctrl+C.

import serial
import numpy as np
from datetime import datetime
from copy import deepcopy
from metric_analysis import AnalysisController
from typing import Dict, List

from data_types import Recording, RecordingEnvironment, Event, WalkPath


def get_samples(columns: int, serial_port: str, baudrate=115200):
    try:
        with serial.Serial(serial_port, baudrate) as ser:
            while True:
                inp = ser.readline()
                try:
                    line = inp.decode().strip().split(',')
                    if len(line) == columns:
                        entries = []
                        for entry in line:
                            try:
                                entries.append(float(entry))
                            except ValueError:
                                entries.append(entry)
                        yield entries
                    else:
                        print(f"Received {len(line)} samples, expected {columns}: {line}")
                except UnicodeDecodeError as e:
                    if inp != '':
                        print(e)
                        yield None
    except KeyboardInterrupt:
        print("Ending recording")


def collect_data(fs: float, serial_port: str, num_sensors=1, error_threshold=0.5):
    start_time = None
    last_time = None
    measurements, events = [], []
    expected_period = 1 / fs
    for line in get_samples(2 + num_sensors, serial_port=serial_port):
        if line is not None:
            timestamp, *samples, event = line
            timestamp = int(timestamp) / 1000
            if last_time is not None and fs is not None:
                fs_error = (timestamp - last_time) - expected_period
                if fs_error >= expected_period * error_threshold:
                    raise EnvironmentError(f'Expected {expected_period} sampling period, received {timestamp - last_time}')
            last_time = timestamp
            samples = [s if isinstance(s, float) else np.nan for s in samples]
            measurements.append(samples)
            if start_time is None:
                start_time = timestamp
            if event == "BUTTON":
                events.append(Event('step', timestamp - start_time))
    measurements = np.array(measurements).T
    return measurements, events


def record_single_test(env: RecordingEnvironment, port='COM8', filename=None):
    measurements, events = collect_data(env.fs, port)
    rec = Recording(env, events, measurements)
    if filename:
        rec.to_file(f'datasets/{filename}.yaml')
    return rec


def record_dual_test(env: RecordingEnvironment, port='COM8', filename=None, accel_rate=100, piezo_rate=200):
    (piezo, accel), events = collect_data(max(accel_rate, piezo_rate), port, num_sensors=2)
    accel = accel[~np.isnan(accel)]
    accel_rec = Recording(deepcopy(env), events, accel)
    accel_rec.env.fs = accel_rate
    if filename:
        accel_rec.to_file(f'datasets/{filename}.yaml')
    piezo = piezo[~np.isnan(piezo)]
    piezo_rec = Recording(deepcopy(env), events, piezo)
    piezo_rec.env.fs = piezo_rate
    if filename:
        piezo_rec.to_file(f'datasets/piezo/{filename}.yaml')
    return accel_rec, piezo_rec


def permute_envs(base_env: RecordingEnvironment, **params: Dict[str, list]):
    """Permute all possible combinations of parameters."""
    from itertools import product
    for vals in product(*params.values()):
        env = deepcopy(base_env)
        for key, val in zip(params.keys(), vals):
            setattr(env, key, val)
        yield env



def sweep_dual_tests(envs: List[RecordingEnvironment], accel_rate=100, piezo_rate=200):
    for env in envs:
        redo = True
        while redo:
            print(env)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            # timestamp = None # Remove this line to save the recordings
            if input("Start recording?") == 'n':
                continue
            accel_rec, piezo_rec = record_dual_test(env, filename=timestamp, accel_rate=accel_rate, piezo_rate=piezo_rate)
            if input("Plot?") != 'n':
                print(AnalysisController(fs=accel_rate).get_recording_metrics(accel_rec, plot=True)[0])
                print(AnalysisController(fs=piezo_rate).get_recording_metrics(piezo_rec, plot=True)[0])
            if input("Redo?") != 'y':
                redo = False


if __name__ == "__main__":
    env = RecordingEnvironment(
        location='Aarons Studio',
        fs=100, # Ignored
        floor='cork',
        obstacle_radius=0,
        wall_radius=0,
        walk_type='normal',
        walk_speed='normal',
        user='ron',
        footwear='socks',
        path=None,
        quality='normal'
    )

    start_dists = [
        2.090840061,
        2.199119678,
        2.342316938,
        2.514473305,
        2.710075541,
        2.924422842,
        3.153695382,
        3.394870578,
        3.645586921,
        3.904006639,
        4.168697370,
        4.438537345,
        4.712642129,
    ]
    paths = [WalkPath(d, d, 4) for d in start_dists]
    qualities = [
        'normal',
        'pause',
        'turn',
        'chaotic',
    ]
    sweep_dual_tests(permute_envs(env, path=paths, quality=qualities))
