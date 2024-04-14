import os
import sys
import traceback
from flask import jsonify, Blueprint, request, send_from_directory
from http import HTTPStatus
from sqlalchemy.exc import OperationalError
import numpy as np
from time import time

from database import db, Recordings, NewSensor, FakeUser
# This is hack, but it's the simplest way to get things to work without changing things - Daniel
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_analysis'))
from data_analysis.metric_analysis import AnalysisController
from data_analysis.data_types import Recording, get_optimal_analysis_params, SensorType
from data_analysis.generate_dummies import generate_metrics, decay


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
    ctrl_params = get_optimal_analysis_params(SensorType.PIEZO, sensor.fs, version=-2, include_model=False)
    analysis_controller = AnalysisController(fs=sensor.fs, noise_amp=0.05, **ctrl_params)
    analysis_controller.get_recording_metrics(rec, plot=True, show=False)
    return analysis_controller.fig.to_html()


@endpoints.route('/api/get_metrics/<email>')
def get_metrics(email: str, fake=True):
    if fake:
        days = 90
        plateau_length = 30
        cadence = np.concatenate([np.array([1.7] * plateau_length), decay(days - plateau_length, 1.7, 1.3)])
        metrics = generate_metrics(days=days, cadence=cadence, asymmetry=0.1, var=0.02, hard_max={'conditional_entropy': 0.2})
        df = metrics.by_tag(smooth_window=7)
        df = df.replace(np.nan, None)
        df.reset_index(inplace=True, drop=False)
        print(df)
        response = jsonify(df.to_dict('list'))
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    try:
        # fs from NewSensor
        # ts_data, date, sensorid from recordings
        user = db.session.query(FakeUser).filter(FakeUser.email == email).first()
        if user is None:
            return jsonify(error=f"User not found: {email}"), HTTPStatus.NOT_FOUND
        #print(user)
        db_userid = user._id
        print("User ID:", db_userid)
        sensor = db.session.query(NewSensor).filter(NewSensor.userid == db_userid).first()
        print("Sample Rate:", sensor.fs)

        recordings = db.session.query(Recordings).filter(Recordings.sensorid == sensor._id).all()
        datasets = [Recording.from_real_data(sensor.fs, recording.ts_data, tag=recording.timestamp.strftime('%Y-%m-%d')) for recording in recordings]
        print("Datasets:", len(datasets))
        if len(datasets) == 0:
            return jsonify(error=f"No recordings found for user: {email}"), HTTPStatus.NOT_FOUND
        start_time = time() # for performance testing
        ctrl_params = get_optimal_analysis_params(SensorType.PIEZO, sensor.fs, version=-2, include_model=False)
        analysis_controller = AnalysisController(fs=sensor.fs, noise_amp=0.05, **ctrl_params)
        metrics = analysis_controller.get_metrics(datasets)[0]
        df = metrics.by_tag()
        df = df.replace(np.nan, None)
        df.reset_index(inplace=True, drop=False)
        time_taken = time() - start_time
        print(time_taken) # will print to server log for performance testing
        print(df)
        response = jsonify(df.to_dict('list'))
        print(response)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        traceback.print_exc()
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
            return jsonify({'message': 'Something broke when querying the FakeUser table!'}), HTTPStatus.INTERNAL_SERVER_ERROR

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


@endpoints.route('/api/create_user', methods=['POST'])
def create_user():
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        sensorid = data.get('sensorid')
        print(sensorid)

        try: # make sure the sensorid is correct!
            db_user_id = db.session.query(NewSensor.userid).filter(NewSensor._id == sensorid).first()
            if db_user_id is None:
                return jsonify({'message': 'Hmmm, are you sure that you have the right SensorId?'}), HTTPStatus.UNAUTHORIZED
        except:
            print ("The query broke! Not ideal.")
            return jsonify({'message': 'Something broke when querying the database!'}), HTTPStatus.INTERNAL_SERVER_ERROR

        userid = db_user_id[0]

        try: # make sure the email hasn't been used already
            db_email = db.session.query(FakeUser.email).filter(FakeUser.email == email).first()
            if db_email is None:
                print("Unique email found. User can proceed.")
            else:
                print("I found this email!")
                print(db_email[0])
                return jsonify({'message': 'Hmmm, it seems you already have an account with that email!'}), HTTPStatus.UNAUTHORIZED

        except:
            print("Something broke!")
            return jsonify({'message': 'Something broke when querying the FakeUser table!'}), HTTPStatus.INTERNAL_SERVER_ERROR


        try: #make sure sensorid hasn't been used before
            db_sensor= db.session.query(FakeUser.sensorid).filter(FakeUser.sensorid == sensorid).first()
            if db_sensor is None:
                print("SensorId is unique. User may proceed.")
            else:
                return jsonify({'message': 'Hmmm, it seems you already have an account with that sensorid!'}), HTTPStatus.UNAUTHORIZED
        except:
            return jsonify({'message': 'Something broke when querying the FakeUser table!'}), HTTPStatus.INTERNAL_SERVER_ERROR

        max_retries = 3

        for attempt in range(max_retries): # retry twice
            try:

                # create database connection
                #database_wakeup()

                new_data = FakeUser(
                    _id=userid, # based on sensorid
                    name=name,
                    email=email,
                    password=password,
                    sensorid=sensorid, 
                )

                db.session.add(new_data)
                db.session.commit() # add to database

                return jsonify({"message": "Data added successfully"}), HTTPStatus.CREATED

            except OperationalError as e:
                print("Operational Error :(")
                db.session.rollback()
                error = e
                return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST
            except Exception as e:
                print("Exception error :(")
                db.session.rollback()
                error = e
                return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST    
            finally:
                print("I'm closing!")
                db.session.close()
    except:         
        return jsonify({"error": str(error)}), HTTPStatus.BAD_REQUEST

@endpoints.route('/', defaults={'path': ''})
@endpoints.route('/<path:path>')
def serve(path):
    if 'api' in path:
        return jsonify(message="Resource not found"), HTTPStatus.NOT_FOUND
    elif path != "" and os.path.exists(STATIC_FOLDER + '/' + path):
        return send_from_directory(STATIC_FOLDER, path)
    else:
        return send_from_directory(STATIC_FOLDER, 'index.html')


