from ultralytics import YOLO

# Load a pre-trained YOLOv8 object detection model
model = YOLO('yolov8n.pt')

# Train the model on your custom dataset using the .yaml file
results = model.train(data='data/custom_data.yaml', epochs=50)
