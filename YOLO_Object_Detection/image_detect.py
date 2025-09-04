from ultralytics import YOLO
import cv2


"""You will get your tranined model after traning inside runs folder"""
model = YOLO("best.pt")  


"""your image name"""
image_path = 'your_image.jpg'  
image = cv2.imread(image_path)

# Check if image is loaded correctly
if image is None:
    print(f"Error: Image '{image_path}' could not be loaded.")
    exit()  

results = model(image)  


boxes = results[0].boxes  
class_names = results[0].names  

for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0].numpy()  
    conf = box.conf[0].item()  
    cls = int(box.cls[0].item())  
    
    print(f"Confidence: {conf}, Class: {class_names[cls]}")

    color = (0, 255, 0) if conf >= 0.8 else (0, 0, 255)  # Green for >= 80%, Red otherwise
    
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    label = f'{class_names[cls]}: {conf:.2f}'
    
    cv2.putText(image, label, (int(x1), int(y1) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

if image is not None and image.shape[0] > 0 and image.shape[1] > 0:
    cv2.imshow('YOLO Object Detection', image)
    cv2.waitKey(0)  
else:
    print("Error: Processed image is empty or invalid.")

cv2.destroyAllWindows()
