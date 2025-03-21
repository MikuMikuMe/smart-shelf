Creating a smart-shelf inventory management system involves using computer vision to track stock levels. Below is a complete Python program that uses OpenCV for computer vision tasks, along with a pre-trained object detection model (such as YOLO or SSD) to identify items. This example assumes a simplified scenario where items can be identified using a camera and known object classes. 

The system will track items on a shelf and send alerts if they fall below a certain threshold. For the alert system, we'll use email notifications. Please make sure to have OpenCV, a pre-trained model, and the necessary packages installed before running this code.

```python
import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load pre-trained object detection model (e.g. YOLO or SSD)
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Define the classes our model can detect
classes = ["item_1", "item_2", "item_3"]  # Customize with your item names

# Set the threshold for low stock
STOCK_THRESHOLD = 2

# Alert system configuration
SENDER_EMAIL = "youremail@example.com"
SENDER_PASSWORD = "yourpassword"
RECEIVER_EMAIL = "receiver@example.com"
SMTP_SERVER = "smtp.example.com"
PORT = 587

def send_email_alert(item):
    """Send email alert for restocking."""
    try:
        subject = f"Restock Alert: {item}"
        body = f"The stock for {item} is below the threshold. Please restock."

        msg = MIMEMultipart()
        msg.attach(MIMEText(body, 'plain'))
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject

        server = smtplib.SMTP(SMTP_SERVER, PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print(f"Alert sent for {item}.")
    except Exception as e:
        print(f"Failed to send alert for {item}. Error: {e}")

def process_frame(frame):
    """Process video frame and determine stock levels."""
    try:
        # Prepare the frame for detection
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Consider only highly confident predictions
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        item_counts = {cls: 0 for cls in classes}

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                item_counts[label] += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check stock and send alerts
        for item, count in item_counts.items():
            if count < STOCK_THRESHOLD:
                print(f"{item} is below threshold: {count} items remaining.")
                send_email_alert(item)

        # Show current status using the video feed (optional)
        cv2.imshow("Frame", frame)

    except Exception as e:
        print(f"Error processing frame: {e}")

def main():
    """Main function to setup video feed and process frames."""
    # Capture video feed from webcam or an IP camera
    cam = cv2.VideoCapture(0)  # Change '0' to the appropriate camera index/id

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            continue

        process_frame(frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### Key Components:
- **Object Detection**: We use YOLOv3 for object detection. You can choose other models like SSD depending on your needs. The configuration and weights files should match the model you choose.
- **Threshold and Alert System**: Each item type is tracked. When stock falls below the `STOCK_THRESHOLD`, an email alert is sent.
- **Error Handling**: Basic error handling is incorporated to capture issues in frame processing and email sending.
- **Dependencies Setup**:
  - Install OpenCV: `pip install opencv-python opencv-python-headless`
  - Set up your email provider's SMTP details for sending alerts.

Adjust paths, email configurations, and item classes as needed. This code is intended as a base template and may require further adjustments depending on specific requirements and environments.