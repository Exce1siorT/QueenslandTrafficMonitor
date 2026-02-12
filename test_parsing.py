#!/usr/bin/env python3
"""
Test script to verify the updated parse_molmo_response function works correctly
"""

from traffic import parse_molmo_response

def test_parsing():
    """Test the updated parsing function with actual molmo8b response format"""
    
    # Sample molmo8b response from the test output
    sample_response = """1) Total vehicle count: 23

2) Detected vehicle bounding boxes:
   [0.543, 415]: Car
   [811 622]: Bus
   [945 479]: Car
   [969 501]: Car
   [975 478]: Car"""
    
    print("Testing parse_molmo_response with actual molmo8b response format...")
    print(f"Response: {sample_response}")
    print()
    
    # Test with typical camera image dimensions (320x256 from the test)
    vehicles_data = parse_molmo_response(sample_response, 320, 256)
    
    print(f"Parsed {len(vehicles_data)} vehicles from molmo8b response:")
    for i, vehicle in enumerate(vehicles_data):
        print(f"  Vehicle {i+1}: x1={vehicle['x1']}, y1={vehicle['y1']}, x2={vehicle['x2']}, y2={vehicle['y2']}")
        print(f"    Width: {vehicle['width']}, Height: {vehicle['height']}")
        print(f"    Center: ({vehicle['center_x']}, {vehicle['center_y']})")
        print()
    
    if len(vehicles_data) > 0:
        print("✅ Parsing successful! Bounding boxes should now appear around detected vehicles.")
    else:
        print("❌ Parsing failed. No vehicles detected.")
    
    return len(vehicles_data) > 0

if __name__ == "__main__":
    success = test_parsing()
    if success:
        print("\n🎉 The bounding box detection fix is working!")
    else:
        print("\n💥 The bounding box detection fix needs more work.")