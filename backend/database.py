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
