import requests
from http import HTTPStatus


# FOR TESTING WITH lOCAL DEPLOYMENT


#local host url endpoint to hit for test sensor post endpoint
url = 'http://localhost:5000/api/sensor_metadata' 

# defines data to send, sends data
def test_event_collection():
    # test data
    data = {
        'sensorid': 2,
        'sampling': 100,
        'floor': 'tile',
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
        if response.status_code == HTTPStatus.CREATED: #was 201, if this breaks its cus i changed this
            print("POST request was successful!")
            print("Response JSON:")
            print(response.json())
        else:
            print("POST request failed with status code:", response.status_code)
            print("Response content:")
            print(response.text)

    return

test_event_collection();










