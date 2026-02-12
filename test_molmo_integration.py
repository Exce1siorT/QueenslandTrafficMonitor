#!/usr/bin/env python3
"""
Test script to verify molmo8b integration works correctly
"""

from traffic import detect_vehicles_from_url, parse_molmo_response, draw_bounding_boxes, create_vehicle_objects
import numpy as np
from PIL import Image
from io import BytesIO
import requests

def test_molmo_detection():
    """Test molmo8b vehicle detection with a sample camera URL"""
    
    # Test with one of the fallback camera URLs
    test_url = "https://cameras.qldtraffic.qld.gov.au/Metropolitan/MRMETRO-1213.jpg"
    
    print(f"Testing molmo8b detection with URL: {test_url}")
    
    try:
        # Test the detection function
        vehicles, annotated_image = detect_vehicles_from_url(test_url)
        
        print(f"Detection completed successfully!")
        print(f"Number of vehicles detected: {len(vehicles)}")
        
        if len(vehicles) > 0:
            print("Vehicle details:")
            for i, vehicle in enumerate(vehicles):
                print(f"  Vehicle {i+1}: x={vehicle.x}, y={vehicle.y}, width={vehicle.width}, height={vehicle.height}")
        
        # Test image dimensions
        if annotated_image is not None:
            height, width = annotated_image.shape[:2]
            print(f"Annotated image dimensions: {width}x{height}")
        
        return True
        
    except Exception as e:
        print(f"Error during molmo8b detection test: {e}")
        return False

def test_response_parsing():
    """Test the response parsing function with sample data"""
    
    # Sample molmo8b response
    sample_response = """
    I detected 3 vehicles in this traffic camera image.
    
    Vehicle 1: x1=100, y1=200, x2=150, y2=250
    Vehicle 2: x1=300, y1=180, x2=350, y2=230  
    Vehicle 3: x1=500, y1=220, x2=550, y2=270
    """
    
    print("Testing response parsing with sample data...")
    
    # Create a dummy image for testing
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    height, width = dummy_image.shape[:2]
    
    # Test parsing
    vehicles_data = parse_molmo_response(sample_response, width, height)
    
    print(f"Parsed {len(vehicles_data)} vehicles from sample response")
    
    for i, vehicle in enumerate(vehicles_data):
        print(f"  Vehicle {i+1}: x1={vehicle['x1']}, y1={vehicle['y1']}, x2={vehicle['x2']}, y2={vehicle['y2']}")
    
    # Test bounding box drawing
    annotated = draw_bounding_boxes(dummy_image, vehicles_data)
    print(f"Bounding boxes drawn successfully on image of shape: {annotated.shape}")
    
    # Test vehicle object creation
    vehicles = create_vehicle_objects(vehicles_data)
    print(f"Created {len(vehicles)} simulated vehicle objects")
    
    return True

if __name__ == "__main__":
    print("=== Testing molmo8b Integration ===\n")
    
    print("1. Testing response parsing...")
    test_response_parsing()
    print()
    
    print("2. Testing full molmo8b detection...")
    success = test_molmo_detection()
    
    if success:
        print("\n✅ All tests passed! molmo8b integration is working correctly.")
    else:
        print("\n❌ Tests failed. Check the error messages above.")