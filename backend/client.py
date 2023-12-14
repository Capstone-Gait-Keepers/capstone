import requests
from http import HTTPStatus
from datetime import datetime

# FOR TESTING WITH lOCAL DEPLOYMENT
#curl -X POST -H "Content-Type: application/json" -d '{"sensorid": 123, "sampling": 5, "floor": "example_floor", "user": "example_user"}' https://capstone-backend-f6qu.onrender.com/api/sensor_metadata


#local host url endpoint to hit for test sensor post endpoint
#url = 'https://capstone-backend-f6qu.onrender.com/api/send_recording' 
#url = 'http://localhost:5000/api/sensor_metadata' 
#url = 'http://localhost:5000/api/sarah_test3' 
#url = 'http://localhost:5000/api/send_recording'
url = 'http://localhost:5000/api/add_sensorconfig'

# defines data to send, sends data
def test_event_collection():
    # test data
    data1 = {
        'sensorid': 47,
        'sampling': 100,
        'floor': 'tile',
        'user': 'julia'
    }

    data2 = "hey"

    data3 = {'text': 'Testing the endpoint with this string.'}
    
    data4 = {
    "text1": 2,
    "text2": 3
    }

    data5 = {
    'sensorid': 2,
    'timestamp': '2023-11-25 03:41:23.295',
    'ts_data': [1.23, 4.56, 7.89]
    }

    data = {
        'model': 'sensor_model_1',
        'fs': 100.5,
        'userid': 123,
        'floor': 'first_floor',
        'wall_radius': 5.0,
        'obstacle_radius': 10
    }

    try:
        response = requests.post(url, json=data)
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
    else:
        # response status
        if response is None:
            print("No response received. The request may have failed.")
        else:
            if response.status_code in (HTTPStatus.OK, HTTPStatus.CREATED): #was 201, if this breaks its cus i changed this
                print("POST request was successful!", response.status_code)
                print(response.content)
            else:
                print("POST request failed with status code:", response.status_code)
                print("Response content:")
                print(response.text)

    return

test_event_collection();










