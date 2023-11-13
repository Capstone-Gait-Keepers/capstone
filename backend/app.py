import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy

# .env
load_dotenv()

app = Flask(__name__)

# database connection
url = os.getenv("DATABSE_URL") # variable from .env
app.config['SQLALCHEMY_DATABASE_URI'] = url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

## DATABASE MODELS

# sensors model
class sensors(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    sampling = db.Column(db.Integer)
    floor = db.Column(db.String(100))
    user = db.Column(db.String(100))

    def __init__(self, sampling, floor, user):
        self.sampling= sampling
        self.floor = floor
        self.user = user
        

db.create_all()

## ENDPOINTS


'''
Proposed interaction to retrieve sensor data:
- sensor posts data as yaml file to backend when it gets a series of steps
OR
- sensor sends data extracted from yaml file to backend when it gets a series of steps
- backend reads the yaml file for sensorId, floor, fs, user, ts data
- ts data goes in a table with timestamps, ts data at timestamps, and sensorId. 
- seperate sensorId table stores the floor, fs, user, ts data information
'''


# retreive sensor metadata
@app.route('/sensor_metadata', methods=['POST'])
def receive_sensor_data():
    data = request.json 
    required_fields = ['sensor_id', 'floor', 'fs', 'user']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    sensor_id = data['sensor_id']
    floor = data['floor']
    fs = data['fs']
    user = data['user']

    new_sensor_data = sensors(
        sensor_id=data['sensor_id'],
        floor=data['floor'],
        fs=data['fs'],
        user=data['user']
    )    

    db.session.add(new_sensor_data)
    db.session.commit()

    return jsonify({'message': 'Sensor data received successfully'}), 200



# Hello World (Daniel)
@app.route('/')
def hello_world():
    return 'Hello, Daniel!'

if __name__ == '__main__':
    app.run(debug=True)