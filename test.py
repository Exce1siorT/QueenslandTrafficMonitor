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

# This initializes the cloud-hosted model once
model = get_model("yolov8n-640")

app = Flask(__name__)

# Using a direct QLD Traffic camera image URL
# This URL returns a refreshed image without changing the address
CAMERA_URL = "https://cameras.qldtraffic.qld.gov.au/Gold_Coast/MRSCHD-293.jpg"

def analyse_traffic(camera_url):
    global _last_result, _last_time
    now = time.time()

    # Only call model at most once per minute
    if _last_result and now - _last_time < 60:
        return _last_result

    try:
        vehicles = detect_vehicles_from_url(camera_url)
        vehicle_count = count_vehicles(vehicles)
        traffic_high = is_traffic_high(vehicle_count)

        _last_result = (vehicle_count, traffic_high)
        _last_time = now
        print(f"Analysis complete: {vehicle_count} vehicles detected, traffic_high={traffic_high}")
        return _last_result
    except Exception as e:
        print(f"Error analyzing traffic: {e}")
        # Return last known result or default values
        return _last_result if _last_result else (0, False)

def detect_vehicles_from_url(image_url):
    # Download the image from the URL
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()
    
    # Convert to PIL Image
    image = Image.open(BytesIO(response.content))
    
    # Convert PIL Image to numpy array (RGB format)
    image_array = np.array(image)
    
    # model.infer() can work with numpy arrays
    results = model.infer(image_array)

    vehicles = []

    # Loop through each result in the list
    for result in results:
        # Each result has a .predictions attribute
        for prediction in result.predictions:
            # Only keep vehicle classes
            if prediction.class_name in ["car", "truck", "bus", "motorcycle"]:
                vehicles.append(prediction)
                print(f"Detected {prediction.class_name} at y={prediction.y:.1f} with confidence {prediction.confidence:.2f}")

    print(f"Total vehicles detected: {len(vehicles)}")
    return vehicles

def count_vehicles(vehicles, min_y=350):
    # Only count objects that appear low enough in the frame
    # Access the y attribute directly from prediction objects
    count = sum(1 for vehicle in vehicles if vehicle.y > min_y)
    print(f"Vehicles below y={min_y}: {count} out of {len(vehicles)}")
    return count

AVERAGE_VEHICLES = 18  # adjust after observing real traffic

def is_traffic_high(vehicle_count):
    threshold = AVERAGE_VEHICLES * 1.3
    print(f"Threshold: {threshold:.1f}, Current: {vehicle_count}")
    return vehicle_count > threshold

@app.route("/")
def index():
    vehicle_count, traffic_high = analyse_traffic(CAMERA_URL)

    return render_template(
        "index.html",
        camera_url=f"{CAMERA_URL}?t={int(time.time())}",
        vehicle_count=vehicle_count,
        traffic_high=traffic_high
    )

if __name__ == "__main__":
    # Debug is enabled to make iteration faster during development
    app.run(debug=True)