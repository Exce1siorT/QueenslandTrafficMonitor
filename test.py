import cv2
import numpy as np
from inference import get_model

# Load YOLOv8 model
model = get_model("yolov8n-640")

# Load original image
image = cv2.imread("test_image.jpg")
orig_h, orig_w = image.shape[:2]

# Run detection - the inference library handles preprocessing
results = model.infer(image)

# Parse detections
predictions = results[0].predictions if hasattr(results[0], 'predictions') else results[0]

for det in predictions:
    # Extract detection info - the inference library returns objects, not arrays
    x1 = det.x - det.width / 2
    y1 = det.y - det.height / 2
    x2 = det.x + det.width / 2
    y2 = det.y + det.height / 2
    conf = det.confidence
    class_name = det.class_name

    # Draw bounding box
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(image, f"{class_name} {conf:.2f}", (int(x1), int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show and save result
cv2.imshow("Detections", image)
cv2.imwrite("detections_output.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()