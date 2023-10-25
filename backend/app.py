from flask import Flask, jsonify, request

app = Flask(__name__)

# sample data
sensor_data = {
    1: {'name': 'Sensor 1', 'value': 1},
    2: {'name': 'Sensor 2', 'value': 2},
}

@app.route('/')
def hello_world():
    return 'Hello, Flask!'

# test api endpoint for sensorId (int) and name (string)
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


if __name__ == '__main__':
    app.run(debug=True)