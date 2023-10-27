print("You're not going crazy")

import requests

url = 'http://localhost:5000/post_example'  # Update the URL as needed

data = {'key': 'value'}

response = None

try:
    response = requests.post(url, json=data)
except requests.exceptions.RequestException as e:
    print(f"Request exception: {e}")

# response status
if response is None:
    print("No response received. The request may have failed.")
else:
    if response.status_code == 200:
        print("POST request was successful!")
        print("Response JSON:")
        print(response.json())
    else:
        print("POST request failed with status code:", response.status_code)
        print("Response content:")
        print(response.text)
    print("Response content: " + response.text)


print("Got to the end")