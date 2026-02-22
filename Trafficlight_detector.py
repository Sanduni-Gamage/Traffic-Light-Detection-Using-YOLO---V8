from ultralytics import YOLO
import cv2
import cvzone

# --- Video Source ---
# For webcam:
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

# For video file:
cap = cv2.VideoCapture("../Videos/trafficlight.mp4")

# --- Load YOLO model ---
model = YOLO("models/trafficlight.pt")
print("Class names in model:", model.names)

# --- Class names from your YAML ---
classNames = ["green", "off", "red", "yellow"]

# Map classes to traffic commands
traffic_cmd = {
    0: "Go",      # green
    2: "Stop",    # red
    3: "Caution"  # yellow
}

while True:
    success, img = cap.read()
    if not success:
        break  # stop if video ends

    # Run detection
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Draw rectangle
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])

            # Skip 'off' lights
            if cls == 1:
                continue

            # Display traffic command instead of raw class
            label = f"{traffic_cmd.get(cls, classNames[cls])} {conf}"

            cvzone.putTextRect(
                img, label, (max(0, x1), max(35, y1)),
                scale=1, thickness=2
            )
    cv2.imshow("Traffic Light Detection", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
