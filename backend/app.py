import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
from http import HTTPStatus

# .env
load_dotenv()

db = SQLAlchemy()

app = Flask(__name__)

# database connection
# url = os.getenv("DATABSE_URL") 
prodpass = os.getenv("PRODPASS") 
prodhost = os.getenv("PRODHOST") 

SQLALCHEMY_DATABASE_URI = f"postgresql://postgres:{prodpass}@{prodhost}:5432/postgres"
#print(SQLALCHEMY_DATABASE_URI)

app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

engine = create_engine(SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)

db.init_app(app)


#print(url)



# DATABASE MODELS FOR CONSISTENT DATA STRUCTURE:

# Use a class to define a table structure.
# Incoming data will be required to fit the class for consistency.
# When the backend gets run, any non-existing tables 
# that have a newly defined class will be (*)should be)
# automatically created in the database.




# sensors model - for sensor metadata
class sensors(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    sampling = db.Column(db.Integer)
    floor = db.Column(db.String(255))
    user = db.Column(db.String(255))




# ENDPOINTS:

# Only existing endpoint right now receives the sensorid,
# the sampling rate of the sensor, the floor type, and 
# the user's name.

# These endpoints are for functionality testing purposes.
# sensor_metadata endpoint works with local deployment.

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
    
# curl -X POST -H "Content-Type:application/json" -d "{'text': 'Testing the endpoint with this string.'}" https://capstone-backend-f6qu.onrender.com/api/sarah_test2
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



# retrieve sensor metadata
# curl --header "Content-Type: application/json" --request POST --data '{"sensorid": "10", "sampling": 100, "floor": "cork", "user": "daniel"}' https://capstone-backend-f6qu.onrender.com/api/sensor_metadata
@app.route('/api/sensor_metadata', methods=['POST'])
def add_data():
    data = request.json()

    try:
        new_data = sensors(
            _id=data['sensorid'],
            sampling=data['sampling'],
            floor=data['floor'],
            user=data['user']
        )

        db.session.add(new_data)
        db.session.commit()

        return jsonify({"message": "Data added successfully"}), HTTPStatus.CREATED

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR


# Proposed interaction to retrieve sensor data:
# - sensor posts data as yaml file to backend when it gets a series of steps
# OR
# - sensor sends data extracted from yaml file to backend when it gets a series of steps
# - backend reads the yaml file for sensorId, floor, fs, user, ts data
# - ts data goes in a table with timestamps, ts data at timestamps, and sensorId. 
# - seperate sensorId table stores the floor, fs, user, ts data information


# Hello World (Daniel)
@app.route('/')
def hello_world():
    return 'Hello, Daniel!'

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        #db.drop_all() #deletes all existing tables
    app.run(debug=True)