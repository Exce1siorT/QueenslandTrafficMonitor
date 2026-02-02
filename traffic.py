from flask import Flask, render_template, send_file, request
from inference import get_model
import time
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import cv2
import threading
from collections import defaultdict

# cache to protect free tier credits
camera_cache = {}
cache_lock = threading.Lock()
model = get_model("yolov8n-640")

app = Flask(__name__)

def load_camera_urls():
    """Load all camera URLs from QLD Traffic API"""
    try:
        response = requests.get(
            "https://api.qldtraffic.qld.gov.au/v1/webcams",
            params={"apikey": "3e83add325cbb69ac4d8e5bf433d770b"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        cameras = []
        for feature in data["features"]:
            camera_info = {
                "url": feature["properties"]["image_url"],
                "location": feature["properties"].get("location_name", "Unknown Location"),
                "camera_id": feature["properties"].get("camera_id", "")
            }
            cameras.append(camera_info)
        
        return cameras
    except Exception as e:
        print(f"Error loading camera URLs: {e}")
        return []

# Load all camera URLs at startup
ALL_CAMERAS = load_camera_urls()
print(f"Loaded {len(ALL_CAMERAS)} cameras")

def analyse_traffic(camera_url):
    """Analyze traffic for a single camera with caching"""
    now = time.time()
    cache_key = camera_url
    
    with cache_lock:
        if cache_key in camera_cache:
            cached_time, result = camera_cache[cache_key]
            if now - cached_time < 60:  # 60 second cache
                return result
    
    try:
        vehicles, annotated_image = detect_vehicles_from_url(camera_url)
        vehicle_count = count_vehicles(vehicles)
        traffic_high = is_traffic_high(vehicle_count)
        
        result = (vehicle_count, traffic_high, annotated_image)
        
        with cache_lock:
            camera_cache[cache_key] = (now, result)
            
        print(f"Analysis complete for {camera_url}: {vehicle_count} vehicles detected, traffic_high = {traffic_high}")
        return result
    except Exception as e:
        print(f"Error analysing traffic for {camera_url}: {e}")
        return (0, False, None)

def detect_vehicles_from_url(image_url):
    # Download the image from the URL
    response = requests.get(image_url, timeout = 10)
    response.raise_for_status()
    
    # Convert to PIL Image
    image = Image.open(BytesIO(response.content))
    
    # Convert PIL Image to numpy array (RGB format)
    image_array = np.array(image)
    
    # Create a copy for drawing bounding boxes
    annotated_image = image_array.copy()
    
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
                #print(f"Detected {prediction.class_name} at y = {prediction.y:.1f} with confidence {prediction.confidence:.2f}")
                
                # Draw bounding box on the annotated image
                x1 = int(prediction.x - prediction.width / 2)
                y1 = int(prediction.y - prediction.height / 2)
                x2 = int(prediction.x + prediction.width / 2)
                y2 = int(prediction.y + prediction.height / 2)
                
                # Draw rectangle
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{prediction.class_name} {prediction.confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"Total vehicles detected: {len(vehicles)}")
    return vehicles, annotated_image

def count_vehicles(vehicles, min_y=0):
    # Count all detected vehicles
    count = sum(1 for vehicle in vehicles if vehicle.y > min_y)
    #print(f"Vehicles counted: {count} out of {len(vehicles)}")
    return count

AVERAGE_VEHICLES = 5  # Lowered for testing - adjust after observing real traffic

def is_traffic_high(vehicle_count):
    threshold = AVERAGE_VEHICLES * 1.3
    #print(f"Threshold: {threshold:.1f}, Current: {vehicle_count}")
    return vehicle_count > threshold

@app.route("/")
def index():
    """Main dashboard showing all cameras"""
    # Process a subset of cameras for performance (e.g., first 12)
    cameras_to_show = ALL_CAMERAS[:12] if ALL_CAMERAS else []
    
    camera_data = []
    for camera in cameras_to_show:
        try:
            vehicle_count, traffic_high, annotated_image = analyse_traffic(camera["url"])
            
            camera_info = {
                "url": camera["url"],
                "location": camera["location"],
                "vehicle_count": vehicle_count,
                "traffic_high": traffic_high,
                "timestamp": int(time.time())
            }
            camera_data.append(camera_info)
        except Exception as e:
            print(f"Error processing camera {camera['url']}: {e}")
            # Add default data for failed cameras
            camera_info = {
                "url": camera["url"],
                "location": camera["location"],
                "vehicle_count": 0,
                "traffic_high": False,
                "timestamp": int(time.time())
            }
            camera_data.append(camera_info)
    
    return render_template("index.html", cameras=camera_data)

@app.route("/annotated_image")
def annotated_image():
    """Serve the annotated image with bounding boxes for a specific camera"""
    camera_url = request.args.get('camera_url')
    
    if not camera_url:
        # Fallback to first camera if no URL provided
        if ALL_CAMERAS:
            camera_url = ALL_CAMERAS[0]["url"]
        else:
            return "No cameras available", 400
    
    # Get cached result for this camera
    cache_key = camera_url
    with cache_lock:
        if cache_key in camera_cache:
            cached_time, result = camera_cache[cache_key]
            vehicle_count, traffic_high, annotated_image = result
            
            if annotated_image is not None:
                # Convert numpy array to PIL Image
                img = Image.fromarray(annotated_image)
                
                # Convert to bytes
                img_io = BytesIO()
                img.save(img_io, 'JPEG', quality=95)
                img_io.seek(0)
                
                return send_file(img_io, mimetype='image/jpeg')
    
    # If no cached image, return original camera image
    try:
        response = requests.get(camera_url, timeout=10)
        return send_file(BytesIO(response.content), mimetype='image/jpeg')
    except Exception as e:
        return f"Error loading image: {e}", 500

if __name__ == "__main__":
    # Debug is enabled to make iteration faster during development
    app.run(debug = True)