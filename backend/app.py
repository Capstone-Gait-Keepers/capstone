from flask import Flask, jsonify, request

app = Flask(__name__)

## SETUP TEST METHODS

# sample data for testing get request
sensor_data = {
    1: {'name': 'Sensor 1', 'value': 1},
    2: {'name': 'Sensor 2', 'value': 2},
}

# Hello World (Daniel)
@app.route('/')
def hello_world():
    return 'Hello, Daniel!'

# POST request example
# works with curl, test.py
@app.route('/post_example', methods=['POST'])
def post_example():
    # get JSON data from the request
    data = request.get_json()

    if data is None:
        return jsonify({"error": "Invalid JSON data"}), 400

    return jsonify({"message": "Received JSON data", "data": data}), 200

# GET request example
# works with curl, test.py
@app.route('/api/sensor_info_test', methods=['GET'])
def get_sensor_data():
    try:
        sensor_id = int(request.args.get('sensorId'))
        name = request.args.get('name')

        if sensor_id not in sensor_data:
            return jsonify({'error': 'Sensor not found'}), 404

        sensor = sensor_data[sensor_id]

        if name and name != sensor['name']:
            return jsonify({'error': 'Sensor name does not match'}), 400

        return jsonify(sensor), 200

    except ValueError:
        return jsonify({'error': 'Invalid sensorId format'}), 400

##

# collects recordingId (step sequence), sensorId (from which sensor), time (time of event), stepTrue(1 for step, 0 for fall)
@app.route('/api/event_collection', methods=['POST'])
def get_events():
    print("Getting events...")
    try:
        data = request.get_json()
        recordingId = data.get('recordingId')
        sensorId = data.get('sensorId')
        timeEvent = data.get('time')
        stepTrue = data.get('stepTrue')

        # error handling
        # if any of these variables are empty / stepTrue != 0 or 1, data is in invalid format
        if not (sensorId and recordingId and timeEvent and stepTrue in (0, 1)):
            return jsonify({'error': 'Invalid data format'}), 400
        
        # data storage
        # key = combo of recordingId and sensorId
        key = f'{recordingId}-{sensorId}'
        
        sensor_data[key] = {
            'time': timeEvent,
            'stepTrue': bool(stepTrue)
        }
        print("Stored data!")

        return jsonify({'message': 'Recording event data created successfully!'}), 201
        
    except:
        return jsonify({'error': 'Invalid data format'}), 400

if __name__ == '__main__':
    app.run(debug=True)