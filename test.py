from flask import Flask, render_template
from inference import get_model
import time
import requests
import numpy as np
from PIL import Image
from io import BytesIO

# cache to protect free tier credits
_last_result = None
_last_time = 0

# Initialising once avoids repeated cold starts of the hosted model
model = get_model("yolov8n-640")

app = Flask(__name__)

def load_camera_urls():
    # Pulling the camera list once avoids repeatedly hitting the public API
    response = requests.get(
        "https://api.qldtraffic.qld.gov.au/v1/webcams",
        params = {"apikey": "3e83add325cbb69ac4d8e5bf433d770b"},
        timeout = 10
    )
    response.raise_for_status()

    data = response.json()

    return [
        feature["properties"]["image_url"]
        for feature in data["features"]
    ]

# Loaded at startup so inference logic stays simple
CAMERA_URLS = load_camera_urls()

def pick_camera():
    # Rotating feeds prevents biasing analysis to one location
    index = int(time.time() / 60) % len(CAMERA_URLS)
    return CAMERA_URLS[index]

def analyse_traffic(camera_url):
    global _last_result, _last_time
    now = time.time()

    # Limiting calls keeps cloud inference costs predictable
    if _last_result and now - _last_time < 60:
        return _last_result

    vehicles = detect_vehicles_from_url(camera_url)
    vehicle_count = count_vehicles(vehicles)
    traffic_high = is_traffic_high(vehicle_count)

    _last_result = (vehicle_count, traffic_high)
    _last_time = now
    return _last_result

def detect_vehicles_from_url(image_url):
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content))
    image_array = np.array(image)

    results = model.infer(image_array)

    vehicles = []

    for result in results:
        for prediction in result.predictions:
            if prediction.class_name in ["car", "truck", "bus", "motorcycle"]:
                vehicles.append(prediction)

    return vehicles

def count_vehicles(vehicles, min_y=350):
    # Filtering by vertical position reduces false positives far from the camera
    return sum(1 for vehicle in vehicles if vehicle.y > min_y)

AVERAGE_VEHICLES = 18

def is_traffic_high(vehicle_count):
    # Relative thresholds adapt better than fixed numbers
    return vehicle_count > AVERAGE_VEHICLES * 1.3

@app.route("/")
def index():
    camera_url = pick_camera()
    vehicle_count, traffic_high = analyse_traffic(camera_url)

    return render_template(
        "index.html",
        camera_url=f"{camera_url}?t={int(time.time())}",
        vehicle_count=vehicle_count,
        traffic_high=traffic_high
    )

if __name__ == "__main__":
    app.run(debug=True)
