import os
import sys
import time
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, func, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship
from http import HTTPStatus
from flask_basicauth import BasicAuth
from datetime import datetime
from sqlalchemy.exc import OperationalError
from flask_sslify import SSLify

# This is hack, but it's the simplest way to get things to work without changing things - Daniel
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_analysis'))
from data_analysis.step_detection import StepDetector
from data_analysis.data_types import Recording, RecordingEnvironment

# .env
load_dotenv()

# init db
db = SQLAlchemy()

app = Flask(__name__, static_folder="static", template_folder="static")

# database connection
url = os.getenv("DATABSE_URL") 
DBUSER = os.getenv("PRODUSER") 
DBID = os.getenv("DB_ID") 
DBPASS = os.getenv("DB_PASS") 
DBREGION = os.getenv("DB_REGION")

SQLALCHEMY_DATABASE_URI = f"postgresql://postgres.{DBID}:{DBPASS}@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
#SQLALCHEMY_DATABASE_URI = f"postgresql://postgres:{prodpass}@{prodhost}:5432/postgres" #old

app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# database init
db.init_app(app)

# SSLify - for TLS security
## forces endpoints to use HTTPS instead of defaulting to HTTP based on client
sslify = SSLify(app)

# BasicAuth configuration
# for documentation page
app.config['BASIC_AUTH_USERNAME'] = os.getenv("DOC_USER")
app.config['BASIC_AUTH_PASSWORD'] = os.getenv("DOC_PASS")
basic_auth = BasicAuth(app)

# max payload size
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

# DATABASE MODELS FOR CONSISTENT DATA STRUCTURE:
## Use a class to define a table structure.
## Incoming data will be required to fit the class for consistency.
## When the backend gets run, any non-existing tables 
## that have a newly defined class will be (*)should be)
## automatically created in the database.

# not in use
class Sensors(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    sampling = db.Column(db.Integer)
    floor = db.Column(db.String(255))
    user = db.Column(db.String(255))

# test model - not in use
class Test(db.Model):
    text1 = db.Column(db.String(255), primary_key=True)
    text2 = db.Column(db.String(255))

# raw recording data
class Recordings(db.Model):
    __tablename__ = 'recordings'
    _id = db.Column("recordingid",db.Integer, primary_key=True)
    sensorid = db.Column(db.Integer, ForeignKey('new_sensor.sensorid'))
    timestamp = db.Column(db.DateTime)
    ts_data = db.Column(db.ARRAY(db.Float))
    new_sensor = relationship('NewSensor', back_populates='recordings')

# for sensor conflig
class NewSensor(db.Model):
    __tablename__ = 'new_sensor'
    _id = db.Column("sensorid", db.Integer, primary_key=True)
    model = db.Column(db.String(255))
    fs = db.Column(db.Float)
    userid = db.Column(db.Integer)
    floor = db.Column(db.String, nullable=True)
    wall_radius = db.Column(db.Float, nullable=True)
    obstacle_radius = db.Column(db.Float, nullable=True)
    recordings = relationship('Recordings', back_populates='new_sensor')

# curl -X POST -d "hey" https://capstone-backend-f6qu.onrender.com/api/sarah_test1
@app.route('/api/sarah_test1', methods=['POST'])
def process_string():
    try:
        raw_data = request.get_data()

        if not raw_data:
            raise ValueError("Empty raw data in the request.")

        return jsonify({'raw_data': raw_data.decode('utf-8')})

    except Exception as e:

        error_message = f"Error processing request: {str(e)}"
        return jsonify({'error': error_message}), HTTPStatus.BAD_REQUEST
    

# curl --header "Content-Type: application/json" --request POST --data '{"text": "10"}' https://capstone-backend-f6qu.onrender.com/api/sarah_test2
@app.route('/api/sarah_test2', methods=['POST'])
def process_json():
    try:
        data = request.get_json()

        if 'text' in data:
            input_string = data['text']
            result = f"Received and processed string: {input_string}"
            return jsonify({'result': result})

        else:
            raise ValueError("'text' key not found in the request JSON data.")

    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        return jsonify({'error': error_message}), HTTPStatus.BAD_REQUEST

# curl --header "Content-Type: application/json" --request POST --data '{"text1": "10", "text2": "100"}' https://capstone-backend-f6qu.onrender.com/api/sarah_test3
@app.route('/api/sarah_test3', methods=['POST'])
def process_json2():
    try:
        data = request.get_json()

        if 'text1' in data and 'text2' in data:
            text1 = data['text1']
            text2 = data['text2']

            result = f"Text1: {text1}, Text2: {text2}"

            return jsonify({'result': result})

        else:
            raise ValueError("Missing 'text1' or 'text2' key in the request JSON data.")

    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        return jsonify({'error': error_message}), HTTPStatus.BAD_REQUEST  

# text1 is a primary key. It will tell you if you made a duplicate entry. Make it something random to avoid that.
# curl --header "Content-Type: application/json" --request POST --data '{"text1": "11", "text2": "100"}' https://capstone-backend-f6qu.onrender.com/api/sarah_test4
# NEED TO INVESTIGATE how tables get created and schema gets updated
@app.route('/api/sarah_test4', methods=['POST'])
def process_json2_withdb():
   data = request.get_json()

   print("Running sarah_test4")

   try:
       new_data = Test(
           text1=data['text1'],
           text2=data['text2'],
       )

       db.session.add(new_data)
       db.session.commit()

       return jsonify({"message": "Data added successfully"}), HTTPStatus.CREATED

   except Exception as e:
       db.session.rollback()
       error_message = f"Error processing request: {str(e)}"
       return jsonify({'error': error_message}), HTTPStatus.BAD_REQUEST  

    
# TO DO: curl 
@app.route('/api/send_recording', methods=['POST'])
def add_recording():
    start_time = time.time() # for performance testing

    data = request.get_json()


    max_retries = 3

    for attempt in range(max_retries): # retry twice
        print("Trying!")
        try:

            print("Wakey Wakey Mr Database!")
            database_wakeup()

            print("Thanks for the data..")
            new_data = Recordings(
                _id=generate_unique_id(), # calls function, populates with value
                sensorid=int(data['sensorid']), # sensor property
                timestamp=datetime.utcnow().isoformat(), # datatime
                ts_data=data['ts_data'], # float 8 array
            )

            print("Ok I'm trying to add data now cross your fingers")

            db.session.add(new_data)
            db.session.commit() # add to database

            print("Omg girls I did it")

            time_taken = time.time() - start_time
            print(time_taken) # will print to server logs for performance testing

            return jsonify({"message": "Data added successfully"}), HTTPStatus.CREATED

        except OperationalError as e:
            print("Operational Error :()")
            db.session.rollback()
            error = e
            #return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST
        except Exception as e:
            print("Exception error :()")
            db.session.rollback()
            error = e
            #return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST    
        finally:
            print("I'm closing!")
            db.session.close()
                
    print("I give up!")
    return jsonify({"error": str(error)}), HTTPStatus.BAD_REQUEST

# for setting up a new sensor
@app.route('/api/add_sensorconfig', methods=['POST'])
def add_sensorconfig():
    start_time = time.time() # for performance testing

    database_wakeup()
    data = request.get_json()
    print(type(data['obstacle_radius']))
    new_sensor_id = generate_unique_id()

    max_retries = 3

    for attempt in range(max_retries): # retry twice
        try:
            new_data = NewSensor(
                _id=new_sensor_id,
                model = str(data['model']),
                fs = float(data['fs']),
                userid = int(data['userid']),
                floor = str(data['floor']),
                wall_radius = float(data['wall_radius']),
                obstacle_radius = float(data['obstacle_radius'])
            )

            db.session.add(new_data)
            db.session.commit() # add to database

            time_taken = time.time() - start_time
            print(time_taken) # will print to server log for performance testing

            return jsonify({"message": "Sensor record created","sensorid": new_sensor_id})
        
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST
        finally:
            db.session.close()



# FUNCTION DEFINITIONS

# generate a unique id to attribute to a recording or to a sensor
def generate_unique_id():
    timestamp = int(time.time() * 1000) 
    unique_id = timestamp % 1000000 #id will be 6 digits long
    return unique_id

def database_wakeup():
    engine = create_engine(SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    #ping = engine.execute("SELECT 1") #pings the database, wakes it up
    time.sleep(0.1) #an idea to try later
    return 

# def query_sensors():
#     query = session.query(NewSensor).all()
#     ids = [result._id for result in query]
#     #print(ids)
#     return ids

# protected by username and password
@app.route('/documentation')
@basic_auth.required
def documentation():
    return render_template('documentation.html')

# shows status of each sensor
@app.route('/api/sensor_status')
def get_sensor_status():

    # create unique session for each query
    engine = create_engine(SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
    connection = engine.connect() 
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        #yield session
        sensors_query = session.query(NewSensor).all()
        ambitious_query = (
            session.query(
                NewSensor._id,
                NewSensor.userid,
                NewSensor.model,
                NewSensor.floor,
                func.count(Recordings.sensorid).label('record_count'),
                func.max(Recordings.timestamp).label('latest_timestamp')
            )
            .outerjoin(Recordings, NewSensor._id == Recordings.sensorid)
            .group_by(NewSensor._id, NewSensor.userid, NewSensor.model, NewSensor.floor)
        )

        sensors = [{'id': new_sensor._id, 'userid': new_sensor.userid, 'model': new_sensor.model, 'floor': new_sensor.floor, 'last_timestamp': new_sensor.latest_timestamp, 'num_recordings': new_sensor.record_count} for new_sensor in ambitious_query]
        
        #handing session instance
        session.commit()

    except Exception as e:
    #exception occurs, rollback the transaction
        db.session.rollback()
        print(f"Error: {str(e)}")
        return "Error occurred, transaction rolled back"
    finally: 
        session.close()

    response = jsonify(sensors)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/api/list_recordings/<int:sensor_id>')
def get_recording_ids(sensor_id: int):
    try:
        recordings = db.session.query(Recordings).filter(Recordings.sensorid == sensor_id).all()
        recording_ids = [recording._id for recording in recordings]
        response = jsonify(recording_ids)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        return jsonify(error=f"Error processing request: {str(e)}"), HTTPStatus.BAD_REQUEST


@app.route('/recording/<int:recording_id>')
def plot_recording(recording_id: int):
    recording = db.session.query(Recordings).filter(Recordings._id == recording_id).first()
    sensor = db.session.query(NewSensor).filter(NewSensor._id == recording.sensorid).first()
    env = RecordingEnvironment(
        sensor._id,
        sensor.fs,
        user='',
        floor=sensor.floor,
        footwear='socks',
        walk_type='normal',
        wall_radius=sensor.wall_radius,
        obstacle_radius=sensor.obstacle_radius
    )
    rec = Recording(env, [], recording.ts_data)
    return rec.plot(show=False).to_html()



@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if 'api' in path:
        return jsonify(message="Resource not found"), HTTPStatus.NOT_FOUND
    elif path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    with app.app_context():
        #print("YAY")
        db.create_all()
        print("Here's the query!")
        #print(query_sensors())
        #db.drop_all() #deletes all existing tables
    app.run(debug=True, ssl_context='adhoc')
