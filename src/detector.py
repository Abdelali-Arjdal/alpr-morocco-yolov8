from ultralytics import YOLO

# Tell the code exactly where to find the model in the repository
model = YOLO('models/best.pt')

def detect_plate(image_path):
    """
    Takes an image path and returns the bounding box of the license plate.
    Output format: [x_min, y_min, x_max, y_max]
    """
    results = model(image_path)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            coordinates = box.xyxy[0].tolist() 
            return coordinates
            
    return None