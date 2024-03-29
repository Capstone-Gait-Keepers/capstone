from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship


db = SQLAlchemy()

# DATABASE MODELS FOR CONSISTENT DATA STRUCTURE:
## Use a class to define a table structure.
## Incoming data will be required to fit the class for consistency.
## When the backend gets run, any non-existing tables 
## that have a newly defined class will be (*)should be)
## automatically created in the database.

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
    users = relationship('FakeUser', back_populates='new_sensor')

# for frontend log in + add user flow
class FakeUser(db.Model):
    __tablename__ = 'users'
    _id = db.Column("userid", db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    email = db.Column(db.String(255))
    password = db.Column(db.String(255))
    usertype = db.Column(db.Integer, nullable=True) # 0 for OA, 1 for caregiver
    sensorid = db.Column(db.Integer, ForeignKey('new_sensor.sensorid'), nullable = True)
    new_sensor = relationship('NewSensor', back_populates='users')