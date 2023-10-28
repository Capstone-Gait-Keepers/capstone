import requests

url = 'http://localhost:5000/api/event_collection' 

# test data
data = {
    'recordingId': 123,
    'sensorId': 1,
    'time': '2023-10-26T14:30:00',
    'stepTrue': 1
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
    if response.status_code == 200 or 201:
        print("POST request was successful!")
        print("Response JSON:")
        print(response.json())
    else:
        print("POST request failed with status code:", response.status_code)
        print("Response content:")
        print(response.text)