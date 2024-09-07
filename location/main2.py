import requests

# Replace 'YOUR_API_KEY' with your actual ipinfo.io API key
API_KEY = '987148be07770f'

# URL to fetch location data for the current IP address
url = f'https://ipinfo.io/json?token={API_KEY}'

response = requests.get(url)
data = response.json()

if response.status_code == 200:
    print(f"IP Address: {data.get('ip')}")
    print(f"City: {data.get('city')}")
    print(f"Region: {data.get('region')}")
    print(f"Country: {data.get('country')}")
    print(f"Location: {data.get('loc')}")
else:
    print(f"Error: {data.get('error', {}).get('info', 'Unknown error')}")
