import requests
import json

url = 'http://localhost:5000/api/sensor_metadata' 
#url = 'http://localhost:5000/api/event_collection' 


def test_event_collection():
    # test data
    data = {
        'sensorid': 1,
        'sampling': 100,
        'floor': 'cork',
        'user': 'ron'
    }

    response = None

    try:
        response = requests.post(url, json=data)
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")

    # response status
    if response is None:
        print("No response received. The request may have failed.")
    else:
        if response.status_code == 201:
            print("POST request was successful!")
            print("Response JSON:")
            print(response.json())
        else:
            print("POST request failed with status code:", response.status_code)
            print("Response content:")
            print(response.text)

    return

def test_create_user():
    data = {
        "name": "Julia"
    }
    json_data = json.dumps(data)
    headers = {'Content-Type': 'application/json'}  

    try:
        response = requests.post(url, data=json_data, headers=headers)
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")

    # response status
    if response.status_code == 201:
        print("Success!")
    elif response.status_code == 400:
        print("Error: Invalid data format")
    else:
        print(f"Unexpected status code: {response.status_code}")
    
    return

test_event_collection();










