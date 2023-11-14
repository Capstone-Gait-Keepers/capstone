import os
from dotenv import load_dotenv
from flask import Flask, jsonify, request, render_template, url_for,redirect,flash
from flask_sqlalchemy import SQLAlchemy # YOU ARE THE DRAMA

# .env
load_dotenv()

app = Flask(__name__)

# database connection
url = os.getenv("DATABSE_URL") # variable from .env
app.config["SQLALCHEMY_DATABASE_URI"] = url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

print(url)


## DATABASE MODELS

# sensors model
class sensors(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    sampling = db.Column(db.Integer)
    floor = db.Column(db.String(255))
    user = db.Column(db.String(255))

## ENDPOINTS

# retrieve sensor metadata
@app.route('/api/sensor_metadata', methods=['POST'])
def add_data():
    data = request.json

    try:
        new_data = sensors(
            _id=data['sensorid'],
            sampling=data['sampling'],
            floor=data['floor'],
            user=data['user']
        )

        db.session.add(new_data)
        db.session.commit()

        return jsonify({"message": "Data added successfully"}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

'''
Proposed interaction to retrieve sensor data:
- sensor posts data as yaml file to backend when it gets a series of steps
OR
- sensor sends data extracted from yaml file to backend when it gets a series of steps
- backend reads the yaml file for sensorId, floor, fs, user, ts data
- ts data goes in a table with timestamps, ts data at timestamps, and sensorId. 
- seperate sensorId table stores the floor, fs, user, ts data information
'''

# Hello World (Daniel)
@app.route('/')
def hello_world():
    return 'Hello, Daniel!'

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        #db.drop_all() deletes all existing tables
    app.run(debug=True)