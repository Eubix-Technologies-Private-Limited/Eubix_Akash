import geocoder
import requests
g = geocoder.ip('me')
g = g.latlng
import requests

def get_location_from_lat_lon(latitude, longitude):
    # Replace 'your_opencage_api_key' with your actual OpenCage API key
    api_key = '40d8153738a04d38866a20949402ca2e'
    reverse_geocode_url = f'https://api.opencagedata.com/geocode/v1/json?q={latitude}+{longitude}&key={api_key}'

    # Make a request to OpenCage API
    response = requests.get(reverse_geocode_url)
    location_data = response.json()

    # Extract location details
    if location_data['results']:
        location_info = location_data['results'][0]
        formatted_address = location_info.get('formatted', 'Address not found')
    else:
        formatted_address = 'Address not found'

    return formatted_address

# Example latitude and longitude
latitude = g[0]
longitude = g[1]

address = get_location_from_lat_lon(latitude, longitude)
print(f"Latitude: {latitude}, Longitude: {longitude}")
print(f"Address: {address}")
'''api_key = '40d8153738a04d38866a20949402ca2e'
reverse_geocode_url = f'https://api.opencagedata.com/geocode/v1/json?q={g[0]}+{g[1]}&key={api_key}'
response = requests.get(reverse_geocode_url)
location_data = response.json()
print(response)
print(location_data)
if location_data['results']:
        location_info = location_data['results'][0]
        formatted_address = location_info.get('formatted', 'Address not found')
else:
    formatted_address = 'Address not found'

print(f"Latitude: {g[0]}, Longitude: {g[1]}")
print(f"Address: {formatted_address}")'''
