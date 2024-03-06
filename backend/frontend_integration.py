import os
import sys
from flask import jsonify, Blueprint, request, send_from_directory
from http import HTTPStatus
from sqlalchemy.exc import OperationalError
from flask_sqlalchemy import SQLAlchemy

#from app import database_wakeup
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
        sensor = db.session.query(FakeUser).join(NewSensor, NewSensor.userid == FakeUser._id).filter(NewSensor.userid == user._id).first() # sampling
        print(sensor)
        datasets = [Recording.from_real_data(sensor.fs, recording.ts_data) for recording in db.session.query(Recordings).filter(Recordings.sensorid == sensor.sensorid).all()]
        print(len(datasets))
        analysis_controller = AnalysisController(fs=sensor.fs, noise_amp=0.05)
        metrics = analysis_controller.get_metrics(datasets)[0]._df.to_dict() #merge moment
        print(metrics)
        response = jsonify(metrics)
        print(response)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify(error=f"Error processing request: {str(e)}"), HTTPStatus.BAD_REQUEST

@endpoints.route('/api/get_user/', methods=['POST'])
def get_user():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # get email and password from db
        try:
            creds = db.session.query(FakeUser).filter(FakeUser.email == email).first()
        except:
            print("Creds didn't work")

        # authenticate
        if creds is None:
            return jsonify({'message': 'Invalid credentials :('}), HTTPStatus.UNAUTHORIZED

        elif email != creds.email: # email is not found
            return jsonify({'message': 'Invalid email :('}), HTTPStatus.UNAUTHORIZED
        
        elif email == creds.email and password == creds.password: # email and password match
            return jsonify({'message': 'Logged in successfully'}), HTTPStatus.OK
        
        elif email == creds.email and password != creds.password: # email is found, but password doesn't match
            return jsonify({'message': 'Invalid password'}), HTTPStatus.UNAUTHORIZED
        
        else: # shouldn't get here, but if they do they were probably unauthorized
            return jsonify({'message': 'Invalid credentials :('}), HTTPStatus.UNAUTHORIZED

    except OperationalError as e:
        print("Operational Error :(")
        db.session.rollback()
        error = e
        #return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST
    except Exception as e:
        print("Exception error :(")
        db.session.rollback()
        error = e
        #return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST    
    finally:
        print("I'm closing!")
        db.session.close()


# @endpoints.route('/api/create_user', methods=['POST'])
# def get_user():
#     try:
#         data = request.get_json()
#         name = data.get('name')
#         email = data.get('email')
#         password = data.get('password')
#         sensorid = data.get('sensorid')

#         userid = 1 # query based on sensorid

#         if userid is None:
#             return jsonify({"message": "I couldn't find that sensorid! Did you enter it correctly?"}), HTTPStatus.UNAUTHORIZED

#         max_retries = 3

#         for attempt in range(max_retries): # retry twice
#             try:

#                 # create database connection
#                 #database_wakeup()

#                 new_data = FakeUser(
#                     _id=userid, # based on sensorid
#                     name=name,
#                     email=email,
#                     password=password,
#                     sensorid=sensorid, 
#                 )

#                 db.session.add(new_data)
#                 db.session.commit() # add to database

#                 return jsonify({"message": "Data added successfully"}), HTTPStatus.CREATED

#             except OperationalError as e:
#                 print("Operational Error :()")
#                 db.session.rollback()
#                 error = e
#                 return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST
#             except Exception as e:
#                 print("Exception error :()")
#                 db.session.rollback()
#                 error = e
#                 return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST    
#             finally:
#                 print("I'm closing!")
#                 db.session.close()
#     except:         
#         return jsonify({"error": str(error)}), HTTPStatus.BAD_REQUEST

@endpoints.route('/', defaults={'path': ''})
@endpoints.route('/<path:path>')
def serve(path):
    if 'api' in path:
        return jsonify(message="Resource not found"), HTTPStatus.NOT_FOUND
    elif path != "" and os.path.exists(STATIC_FOLDER + '/' + path):
        return send_from_directory(STATIC_FOLDER, path)
    else:
        return send_from_directory(STATIC_FOLDER, 'index.html')

