import requests

API_KEY = 'd53cd3ec5e596cc2026f9a7765dbaaab'
url = f'https://www.googleapis.com/geolocation/v1/geolocate?key={API_KEY}'

response = requests.post(url)
data = response.json()

'''latitude = data['location']['lat']
longitude = data['location']['lng']

print(f"Latitude: {latitude}")
print(f"Longitude: {longitude}")'''
print(data)
