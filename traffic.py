from flask import Flask, render_template
from inference import get_model
import time

# cache to protect free tier credits
_last_result = None
_last_time = 0

#test comment

# This initializes the cloud-hosted model once
model = get_model("yolov8n-640")

app = Flask(__name__)

# Using a direct QLD Traffic camera image URL
# This URL returns a refreshed image without changing the address
CAMERA_URL = "https://cameras.qldtraffic.qld.gov.au/Gold_Coast/MRSCHD-293.jpg"

def fetch_camera_image(url):
    # Returning None for now lets the pipeline exist without dependencies
    return None

def analyse_traffic(camera_url):
    global last_result, _last_time
    now = time.time()

    # Only call Roboflow at most once per minute
    if last_result and now - last_time < 60:
        return last_result

    vehicles = detect_vehicles_from_url(camera_url)
    vehicle_count = count_vehicles(vehicles)
    traffic_high = is_traffic_high(vehicle_count)

    last_result = (vehicle_count, traffic_high)
    last_time = now
    return last_result

def detect_vehicles_from_url(image_url):
    # model.infer() returns a list of ObjectDetectionInferenceResponse objects
    results = model.infer(image_url)

    vehicles = []

    # Loop through each result in the list
    for result in results:
        # Each res has a .predictions attribute
        for prediction in result.predictions:
            # Only keep vehicle classes
            if prediction.class_name in ["car", "truck", "bus", "motorcycle"]:
                vehicles.append(prediction)

    return vehicles

def count_vehicles(vehicles, min_y = 350):
    # Only count objects that appear low enough in the frame
    return sum(1 for vehicle in vehicles if vehicle["y"] > min_y)

AVERAGE_VEHICLES = 18  # adjust after observing real traffic

def is_traffic_high(vehicle_count):
    return vehicle_count > AVERAGE_VEHICLES * 1.3

@app.route("/")
def index():
    vehicle_count, traffic_high = analyse_traffic(CAMERA_URL)

    return render_template(
        "index.html",
        camera_url = f"{CAMERA_URL}?t={int(time.time())}",
        vehicle_count = vehicle_count,
        traffic_high = traffic_high
    )

if __name__ == "__main__":
    # Debug is enabled to make iteration faster during development
    app.run(debug = True)
