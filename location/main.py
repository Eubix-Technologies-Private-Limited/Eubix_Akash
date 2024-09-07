'''import requests

response = requests.get('https://ipinfo.io/json')
data = response.json()
print(data)

print(f"IP: {data['ip']}")
print(f"Location: {data['city']}, {data['region']}, {data['country']}")
print(f"Latitude and Longitude: {data['loc']}")
'''
import requests

# Replace 'YOUR_API_KEY' with your actual IPstack API key
API_KEY = 'd53cd3ec5e596cc2026f9a7765dbaaab'

url = f'http://api.ipstack.com/check?access_key={API_KEY}'

response = requests.get(url)
data = response.json()
print(data)
if response.status_code == 200:
    print(f"IP Address: {data.get('ip')}")
    print(f"City: {data.get('city')}")
    print(f"Region: {data.get('region_name')}")
    print(f"Country: {data.get('country_name')}")
    print(f"Latitude: {data.get('latitude')}")
    print(f"Longitude: {data.get('longitude')}")
else:
    print(f"Error: {data.get('error', {}).get('info', 'Unknown error')}")
