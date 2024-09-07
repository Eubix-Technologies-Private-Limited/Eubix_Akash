'''from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/location', methods=['POST'])
def get_location():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    print(f"Latitude: {latitude}, Longitude: {longitude}")
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
'''
'''from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/location', methods=['POST'])
def get_location():
    data = request.get_json()
    print(data)
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    print(f"Received Location: Latitude: {latitude}, Longitude: {longitude}")
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
'''
from flask import Flask, request, jsonify, send_from_directory
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/location', methods=['POST'])
def get_location():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    accuracy = data.get('accuracy')

    # Replace 'your_opencage_api_key' with your actual OpenCage API key
    api_key = '40d8153738a04d38866a20949402ca2e'
    reverse_geocode_url = f'https://api.opencagedata.com/geocode/v1/json?q={latitude}+{longitude}&key={api_key}'

    # Make a request to OpenCage API
    response = requests.get(reverse_geocode_url)
    location_data = response.json()
    #print(response)
    #print(location_data)

    # Extract location details
    if location_data['results']:
        location_info = location_data['results'][0]
        formatted_address = location_info.get('formatted', 'Address not found')
    else:
        formatted_address = 'Address not found'

    print(f"Latitude: {latitude}, Longitude: {longitude}, Accuracy: {accuracy} meters")
    print(f"Address: {formatted_address}")

    return jsonify({"status": "success", "address": formatted_address})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
