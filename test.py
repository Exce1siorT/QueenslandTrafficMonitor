import cv2
import numpy as np
from inference import get_model

# Load YOLOv8 model
model = get_model("yolov8n-640")

# Load original image
image = cv2.imread("test_image.jpg")
orig_h, orig_w = image.shape[:2]

# Convert BGR -> RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize while keeping aspect ratio
input_size = 640
scale = min(input_size / orig_w, input_size / orig_h)
new_w = int(orig_w * scale)
new_h = int(orig_h * scale)
resized = cv2.resize(image_rgb, (new_w, new_h))

# Pad to make square (640x640)
pad_w = input_size - new_w
pad_h = input_size - new_h
top = pad_h // 2
bottom = pad_h - top
left = pad_w // 2
right = pad_w - left
padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

# Convert to float32 and normalise
padded = padded.astype(np.float32) / 255.0

# Channels first and batch dimension
input_tensor = np.expand_dims(np.transpose(padded, (2, 0, 1)), axis=0)

# Run detection
results = model.predict(input_tensor)

# Parse detections
class_names = model.class_names
detections = results[0]

for det in detections:
    # Take first 6 elements
    x1, y1, x2, y2, conf, class_id = det[:6]

    # Convert box coordinates back to original image
    x1 = (x1 - left) / scale
    x2 = (x2 - left) / scale
    y1 = (y1 - top) / scale
    y2 = (y2 - top) / scale

    # Ensure class_id is an integer
    class_name = class_names[int(np.round(class_id).item())]

    # Draw bounding box
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, f"{class_name} {conf:.2f}", (int(x1), int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show and save result
cv2.imshow("Detections", image)
cv2.imwrite("detections_output.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
