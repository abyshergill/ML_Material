from ultralytics import YOLO
import cv2

"""You will get your tranined model after traning inside runs folder"""
model = YOLO("best.pt")  

"""Your Video """
video_path = 'your_video.mp4'  
cap = cv2.VideoCapture(video_path)   

while cap.isOpened():  
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes  
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  
            conf = box.conf[0] 
            cls = box.cls[0]    
            
            color = (0, 255, 0) if conf >= 0.8 else (0, 0, 255)  # Green for >= 80%, Red otherwise
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f'{result.names[int(cls)]}: {conf:.2f}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Aby Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

