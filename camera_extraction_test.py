import requests

url = "https://api.qldtraffic.qld.gov.au/v1/webcams"
params = {
    "apikey": "3e83add325cbb69ac4d8e5bf433d770b"  # This is required, even for the public key
}

response = requests.get(url, params=params)

# This prevents assuming the response shape when the request failed
print("Status:", response.status_code)
print(response.text)

response.raise_for_status()

data = response.json()

for feature in data["features"]:
    print(feature["properties"]["image_url"])
