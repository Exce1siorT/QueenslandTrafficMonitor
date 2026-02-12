from flask import Flask, render_template, send_file, request
from openai import OpenAI
import time
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import cv2
import threading
from collections import defaultdict
import re
import os

# cache to protect free tier credits
camera_cache = {}
cache_lock = threading.Lock()

def extract_location_from_url(image_url):
    """
    Extract location information from camera image URL
    
    :param image_url: URL of the traffic camera image
    :return: Extracted location name from filename
    """
    try:
        # Extract the filename from the URL
        filename = os.path.basename(image_url)
        
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Try to extract meaningful location from filename patterns
        # Common patterns in QLD traffic camera URLs:
        # - MRMETRO-1213.jpg (Metropolitan camera)
        # - BRISBANE-001.jpg (Brisbane camera)
        # - GOLD-COAST-005.jpg (Gold Coast camera)
        
        # Remove common prefixes and suffixes
        location = name_without_ext
        
        # Handle common camera ID patterns
        # Remove trailing numbers if they're just camera IDs
        location = re.sub(r'-\d+$', '', location)
        location = re.sub(r'_\d+$', '', location)
        
        # Replace hyphens and underscores with spaces for readability
        location = location.replace('-', ' ').replace('_', ' ')
        
        # Capitalize words
        location = ' '.join(word.capitalize() for word in location.split())
        
        # If the result is too short or generic, try to extract from directory path
        if len(location) < 3 or location.lower() in ['camera', 'image', 'cam']:
            # Extract from the directory path
            path_parts = image_url.split('/')
            # Look for meaningful location names in the path
            for part in reversed(path_parts[:-1]):  # Skip the filename itself
                if part and len(part) > 2 and not part.isdigit():
                    location = part.replace('-', ' ').replace('_', ' ')
                    location = ' '.join(word.capitalize() for word in location.split())
                    break
        
        return location if location else "Unknown Location"
        
    except Exception as e:
        print(f"Error extracting location from URL {image_url}: {e}")
        return "Unknown Location"

# Initialize OpenAI client for molmo8b
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

AVERAGE_VEHICLES = 5  # Lowered for testing - adjust after observing real traffic

app = Flask(__name__)

def load_camera_urls():
    """
    Load all camera URLs from QLD Traffic API
    
    returns: cameras
    """
    try:
        # Use a more specific API call to avoid rate limiting
        response = requests.get(
            "https://api.qldtraffic.qld.gov.au/v1/webcams",
            params={
                "apikey": "3e83add325cbb69ac4d8e5bf433d770b",
                "limit": 20  # Limit the number of cameras to avoid rate limits
            },
            timeout=15
        )
        response.raise_for_status()
        data = response.json()
        
        cameras = []
        for feature in data["features"]:
            camera_info = {
                "url": feature["properties"]["image_url"],
                "location": extract_location_from_url(feature["properties"]["image_url"]),
                "camera_id": feature["properties"].get("camera_id", "")
            }
            cameras.append(camera_info)
        
        print(f"Successfully loaded {len(cameras)} cameras")
        return cameras
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print("Rate limit exceeded. Using fallback camera URLs.")
            # Return some sample camera URLs for testing
            return [
                {
                    "url": "https://cameras.qldtraffic.qld.gov.au/Metropolitan/MRMETRO-1213.jpg",
                    "location": "Sample Camera 1",
                    "camera_id": "sample1"
                },
                {
                    "url": "https://cameras.qldtraffic.qld.gov.au/Metropolitan/MRMETRO-1214.jpg", 
                    "location": "Sample Camera 2",
                    "camera_id": "sample2"
                }
            ]
        else:
            print(f"HTTP Error loading camera URLs: {e}")
            return []
    except Exception as e:
        print(f"Error loading camera URLs: {e}")
        return []


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
        vehicle_count = detect_vehicles_from_url(camera_url)
        traffic_high = is_traffic_high(vehicle_count)
        
        result = (vehicle_count, traffic_high, None)  # No annotated image needed
        
        with cache_lock:
            camera_cache[cache_key] = (now, result)
            
        print(f"Analysis complete for {camera_url}: {vehicle_count} vehicles detected, traffic_high = {traffic_high}")
        return result
    except Exception as e:
        print(f"Error analysing traffic for {camera_url}: {e}")
        return (0, False, None)

def detect_vehicles_from_url(image_url):
    """
    Detect vehicles using molmo8b model for counting only
    
    :param image_url: URL of the traffic camera image
    
    returns: vehicle count (integer)
    """
    try:
        # Use molmo8b to count vehicles only
        completion = client.chat.completions.create(
            model="allenai/molmo-2-8b",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Count the number of vehicles in this traffic camera image. Only return the number as a single integer. Do not provide any explanations or additional text."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                    ]
                }
            ]
        )
        
        # Extract just the number from the response
        ai_response = completion.choices[0].message.content
        print(f"molmo8b response: {ai_response}")
        
        # Extract the number from response text
        import re
        numbers = re.findall(r'\d+', ai_response)
        if numbers:
            vehicle_count = int(numbers[-1])  # Use the last number found
        else:
            vehicle_count = 0
            
        print(f"Vehicles counted by molmo8b: {vehicle_count}")
        return vehicle_count
        
    except Exception as e:
        print(f"Error using molmo8b: {e}")
        return 0

def parse_molmo_response(response_text, image_width, image_height):
    """
    Parse molmo8b response to extract vehicle bounding box coordinates
    
    :param response_text: Text response from molmo8b
    :param image_width: Width of the image
    :param image_height: Height of the image
    :return: List of vehicle data with coordinates
    """
    vehicles_data = []
    
    # Clean the response text to handle various formats
    cleaned_text = response_text.replace('\n', ' ').replace('\r', ' ')
    
    # Extract coordinate patterns - handle the actual molmo8b response format
    # molmo8b typically returns coordinates like: [0.543, 415]: Car or [811 622]: Bus
    coord_patterns = [
        # Handle bracket format with colon and space: [x, y]: Type or [x y]: Type
        r'\[(\d+\.?\d*),?\s*(\d+\.?\d*)\]:\s*\w+',
        # Handle bracket format with just space: [x y]: Type
        r'\[(\d+\.?\d*)\s+(\d+\.?\d*)\]:\s*\w+',
        # Handle bracket format without colon: [x, y] Type or [x y] Type
        r'\[(\d+\.?\d*),?\s*(\d+\.?\d*)\]\s+\w+',
        # Handle bracket format with dash: [x, y] - Type
        r'\[(\d+\.?\d*),?\s*(\d+\.?\d*)\]\s*-\s*\w+',
        # Handle simple bracket format: [x, y]
        r'\[(\d+\.?\d*),?\s*(\d+\.?\d*)\]',
        # Handle space-separated coordinates: x y
        r'(\d+\.?\d*)\s+(\d+\.?\d*)',
        # Handle comma-separated coordinates: x, y
        r'(\d+\.?\d*),\s*(\d+\.?\d*)',
        # Handle decimal format with colons or equals (legacy support)
        r'x1[:=]\s*(\d+\.?\d*),?\s*y1[:=]\s*(\d+\.?\d*),?\s*x2[:=]\s*(\d+\.?\d*),?\s*y2[:=]\s*(\d+\.?\d*)',
        # Handle bracket format like [0.051,0.244], [0.212,0.475] (legacy support)
        r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\],?\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)\]',
        # Handle coordinates with labels like "Vehicle 1: 105,273,156,368" (legacy support)
        r'Vehicle\s+\d+.*?(\d+),(\d+),(\d+),(\d+)',
    ]
    
    for pattern in coord_patterns:
        matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
        for match in matches:
            if len(match) >= 2:  # We need at least 2 coordinates (x, y)
                try:
                    # Handle different coordinate formats
                    if len(match) == 2:
                        # Single coordinate pair (likely center point or single corner)
                        x, y = map(float, match)
                        
                        # For single points, create a small bounding box around the point
                        # This is common when molmo8b returns single coordinates
                        box_size = min(50, image_width // 20, image_height // 20)  # Adaptive box size
                        x1 = max(0, int(x - box_size // 2))
                        y1 = max(0, int(y - box_size // 2))
                        x2 = min(image_width, int(x + box_size // 2))
                        y2 = min(image_height, int(y + box_size // 2))
                        
                    elif len(match) == 4:
                        # Four coordinates (likely x1,y1,x2,y2 or two coordinate pairs)
                        coords = list(map(float, match))
                        
                        # Determine if it's x1,y1,x2,y2 or x1,x2,y1,y2 by checking ranges
                        if coords[0] < coords[2] and coords[1] < coords[3]:
                            # Standard x1,y1,x2,y2 format
                            x1, y1, x2, y2 = map(int, coords)
                        elif coords[0] < image_width and coords[1] < image_height:
                            # Likely x1,y1,x2,y2 but need to validate
                            x1, y1, x2, y2 = map(int, coords)
                        else:
                            # Try to interpret as two coordinate pairs
                            x1, x2, y1, y2 = map(int, coords)
                            if x1 > x2: x1, x2 = x2, x1
                            if y1 > y2: y1, y2 = y2, y1
                    
                    else:
                        # More than 4 coordinates, take first 4
                        x1, y1, x2, y2 = map(int, match[:4])
                    
                    # Validate coordinates are within image bounds and make sense
                    if (0 <= x1 < x2 <= image_width and 
                        0 <= y1 < y2 <= image_height and
                        x2 - x1 > 5 and y2 - y1 > 5):  # Minimum size check (reduced from 10)
                        
                        vehicle_data = {
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'center_x': (x1 + x2) // 2,
                            'center_y': (y1 + y2) // 2
                        }
                        vehicles_data.append(vehicle_data)
                        
                        # Limit to prevent too many detections from parsing errors
                        if len(vehicles_data) >= 50:  # Reasonable limit for traffic cameras
                            break
                            
                except (ValueError, IndexError):
                    continue
        
        # If we found vehicles with this pattern, don't try other patterns
        if vehicles_data:
            break
    
    # If no coordinates found with bracket patterns, try to extract any numbers that could be coordinates
    if not vehicles_data:
        numbers = re.findall(r'\d+', cleaned_text)
        # Try to group numbers into sets of 2 (x,y pairs) or 4 (x1,y1,x2,y2)
        for i in range(0, len(numbers) - 1, 2):
            try:
                x, y = map(int, numbers[i:i+2])
                # Create a small bounding box around the point
                box_size = min(50, image_width // 20, image_height // 20)
                x1 = max(0, x - box_size // 2)
                y1 = max(0, y - box_size // 2)
                x2 = min(image_width, x + box_size // 2)
                y2 = min(image_height, y + box_size // 2)
                
                if (0 <= x1 < x2 <= image_width and 
                    0 <= y1 < y2 <= image_height and
                    x2 - x1 > 5 and y2 - y1 > 5):
                    vehicle_data = {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'center_x': (x1 + x2) // 2,
                        'center_y': (y1 + y2) // 2
                    }
                    vehicles_data.append(vehicle_data)
                    
                    if len(vehicles_data) >= 50:
                        break
            except (ValueError, IndexError):
                continue
    
    return vehicles_data

def draw_bounding_boxes(image_array, vehicles_data):
    """
    Draw bounding boxes on the image
    
    :param image_array: Numpy array of the image
    :param vehicles_data: List of vehicle data with coordinates
    :return: Image array with bounding boxes drawn
    """
    annotated_image = image_array.copy()
    
    for i, vehicle in enumerate(vehicles_data):
        x1, y1, x2, y2 = vehicle['x1'], vehicle['y1'], vehicle['x2'], vehicle['y2']
        
        # Draw rectangle
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label with vehicle number
        label = f"Vehicle {i+1}"
        cv2.putText(annotated_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_image

def create_vehicle_objects(vehicles_data):
    """
    Create simulated vehicle objects for compatibility with existing YOLO code
    
    :param vehicles_data: List of vehicle data with coordinates
    :return: List of simulated vehicle objects
    """
    class SimulatedVehicle:
        def __init__(self, data):
            self.x = data['center_x']
            self.y = data['center_y']
            self.width = data['width']
            self.height = data['height']
            self.class_name = "vehicle"  # Generic vehicle class
            self.confidence = 0.9  # Default confidence for molmo8b
    
    vehicles = []
    for vehicle_data in vehicles_data:
        vehicle = SimulatedVehicle(vehicle_data)
        vehicles.append(vehicle)
    
    return vehicles

def count_vehicles(vehicles, min_y = 0):
    # Count all detected vehicles
    count = sum(1 for vehicle in vehicles if vehicle.y > min_y)
    #print(f"Vehicles counted: {count} out of {len(vehicles)}")
    return count


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
    
    return render_template("index.html", cameras = camera_data)


if __name__ == "__main__":
    # Load all camera URLs at startup
    ALL_CAMERAS = load_camera_urls()
    print(f"Loaded {len(ALL_CAMERAS)} cameras")

    # Debug is enabled to make iteration faster during development
    app.run(debug = True)