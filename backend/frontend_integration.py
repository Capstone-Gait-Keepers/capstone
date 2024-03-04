import os
import sys
from flask import jsonify, Blueprint, send_from_directory
from http import HTTPStatus


from database import db, Recordings, NewSensor, FakeUser
# This is hack, but it's the simplest way to get things to work without changing things - Daniel
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_analysis'))
from data_analysis.metric_analysis import AnalysisController
from data_analysis.data_types import Recording


STATIC_FOLDER = "static"
endpoints = Blueprint('endpoints', __name__, template_folder=STATIC_FOLDER)
# piezo_model = Recording.from_file('ctrl_model.yaml')



@endpoints.route('/api/list_recordings/<int:sensor_id>')
def get_recording_ids(sensor_id: int):
    try:
        recordings = db.session.query(Recordings).filter(Recordings.sensorid == sensor_id).all()
        recording_ids = [recording._id for recording in recordings]
        response = jsonify(recording_ids)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify(error=f"Error processing request: {str(e)}"), HTTPStatus.BAD_REQUEST


@endpoints.route('/recording/<int:recording_id>')
def plot_recording(recording_id: int):
    recording = db.session.query(Recordings).filter(Recordings._id == recording_id).first()
    sensor = db.session.query(NewSensor).filter(NewSensor._id == recording.sensorid).first()
    rec = Recording.from_real_data(sensor.fs, recording.ts_data)
    return rec.plot(show=False).to_html()


@endpoints.route('/api/get_metrics/<email>')
def get_metrics(email: str):
    try:
        # fs from NewSensor
        # ts_data, date, sensorid from recordings
        print(email)
        user = db.session.query(FakeUser).filter(FakeUser.email == email).first()
        sensor = db.session.query(FakeUser).filter(NewSensor.userid == user._id).first() # sampling
        # user = db.session.query(NewSensor).filter(FakeUser.email == email).first()
        sensor = db.session.query(NewSensor).filter(NewSensor.userid == user._id).first()
        print(sensor)
        datasets = [Recording.from_real_data(sensor.fs, recording.ts_data) for recording in db.session.query(Recordings).filter(Recordings.email == email).all()]
        print(len(datasets))
        analysis_controller = AnalysisController(fs=200, noise_amp=0.05)
        metrics = analysis_controller.get_metrics(datasets)[0]._df.to_dict()
        print(metrics)
        response = jsonify(metrics)
        print(response)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify(error=f"Error processing request: {str(e)}"), HTTPStatus.BAD_REQUEST


@endpoints.route('/', defaults={'path': ''})
@endpoints.route('/<path:path>')
def serve(path):
    if 'api' in path:
        return jsonify(message="Resource not found"), HTTPStatus.NOT_FOUND
    elif path != "" and os.path.exists(STATIC_FOLDER + '/' + path):
        return send_from_directory(STATIC_FOLDER, path)
    else:
        return send_from_directory(STATIC_FOLDER, 'index.html')
