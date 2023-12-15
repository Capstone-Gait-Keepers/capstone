import os
import time
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from http import HTTPStatus
from flask_basicauth import BasicAuth

# .env
load_dotenv()

# init
db = SQLAlchemy()

app = Flask(__name__)

# database connection
url = os.getenv("DATABSE_URL") 
prodpass = os.getenv("PRODPASS") 
prodhost = os.getenv("PRODHOST") 
SQLALCHEMY_DATABASE_URI = f"postgresql://postgres:{prodpass}@{prodhost}:5432/postgres"
#print(SQLALCHEMY_DATABASE_URI)
app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

engine = create_engine(SQLALCHEMY_DATABASE_URI)
connection = engine.connect() 
Session = sessionmaker(bind=engine)
session = Session()


db.init_app(app)

# BasicAuth configuration
# for documentation page
app.config['BASIC_AUTH_USERNAME'] = os.getenv("DOC_USER")
app.config['BASIC_AUTH_PASSWORD'] = os.getenv("DOC_PASS")
basic_auth = BasicAuth(app)

# DATABASE MODELS FOR CONSISTENT DATA STRUCTURE:

# Use a class to define a table structure.
# Incoming data will be required to fit the class for consistency.
# When the backend gets run, any non-existing tables 
# that have a newly defined class will be (*)should be)
# automatically created in the database.


# sensors model - for sensor metadata
class Sensors(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    sampling = db.Column(db.Integer)
    floor = db.Column(db.String(255))
    user = db.Column(db.String(255))

# test model
class Test(db.Model):
    text1 = db.Column(db.String(255), primary_key=True)
    text2 = db.Column(db.String(255))

# raw recording data
class Recordings(db.Model):
    _id = db.Column("recordingid",db.Integer, primary_key=True)
    sensorid = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime)
    ts_data = db.Column(db.ARRAY(db.Float))

class NewSensor(db.Model):
    _id = db.Column("sensorid", db.Integer, primary_key=True)
    model = db.Column(db.String(255))
    fs = db.Column(db.Float)
    userid = db.Column(db.Integer)
    floor = db.Column(db.String)
    wall_radius = db.Column(db.Float)
    obstacle_radius = db.Column(db.Float)

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
    data = request.get_json()
    try:
        new_data = Recordings(
            _id=generate_unique_id(), # calls function, populates with value
            sensorid=int(data['sensorid']), # sensor property
            timestamp=data['timestamp'], # datatime
            ts_data=data['ts_data'], # float 8 array
        )

        db.session.add(new_data)
        db.session.commit() # add to database

        return jsonify({"message": "Data added successfully"}), HTTPStatus.CREATED

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST


# for setting up a new sensor
@app.route('/api/add_sensorconfig', methods=['POST'])
def add_sensorconfig():
    data = request.get_json()
    print(type(data['obstacle_radius']))
    new_sensor_id = generate_unique_id()
    
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

        return jsonify({"message": "Sensor record created","sensorid": new_sensor_id})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), HTTPStatus.BAD_REQUEST




# FUNCTION DEFINITIONS

# generate a unique id to attribute to a recording or to a sensor
def generate_unique_id():
    timestamp = int(time.time() * 1000) 
    unique_id = timestamp % 1000000 #id will be 6 digits long
    return unique_id

def query_sensors():
    query = session.query(NewSensor).all()
    ids = [result._id for result in query]
    #print(ids)
    return ids

# Hello World (Daniel)
@app.route('/')
def hello_world():
    return render_template('hello.html')

# protected by username and password
@app.route('/documentation')
@basic_auth.required
def documentation():
    return render_template('documentation.html')

# shows status of each sensor
@app.route('/status')
def sensor_page():
    sensors_query = session.query(NewSensor).all()

    print("HELLLOO I DONT BELIEVE YOU JULIA (WELL, NOT CONFIDENT)")

    sensors = [{'id': new_sensor._id, 'userid': new_sensor.userid, 'model': new_sensor.model, 'floor': new_sensor.floor, 'last_timestamp': '2023-01-01 12:00:00', 'num_recordings': 10} for new_sensor in sensors_query]
    sensor_ids = [sensor._id for sensor in sensors_query]

    #sensors_query = session.query(Recordings).where(f"sensor_id in {sensor_ids}")

    # sensors = [
    #     {'id': 1, 'userid': 1, 'last_timestamp': '2023-01-01 12:00:00', 'num_recordings': 10},
    #     {'id': 2, 'last_timestamp': '2023-01-01 12:15:00', 'num_recordings': 8},
    #     {'id': 3, 'last_timestamp': '2023-01-01 12:30:00', 'num_recordings': 6},
    #     {'id': 4, 'last_timestamp': '2023-01-01 12:45:00', 'num_recordings': 4},
    #     {'id': 5, 'last_timestamp': '2023-01-01 13:00:00', 'num_recordings': 2},
    # ]

    return render_template('status.html', sensors=sensors)


if __name__ == '__main__':
    with app.app_context():
        #print("YAY")
        db.create_all()
        print("Here's the query!")
        print(query_sensors())
        #db.drop_all() #deletes all existing tables
    app.run(debug=True)
