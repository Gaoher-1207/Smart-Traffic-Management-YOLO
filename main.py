from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture(0)

# Vehicle classes
vehicle_classes = ["car", "motorcycle", "bus", "truck"]

# Create window once
cv2.namedWindow("Traffic Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Detection", 1100, 650)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    vehicle_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name in vehicle_classes:
            vehicle_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Traffic Density Logic
    if vehicle_count < 8:
        density = "LOW"
        color = (0, 255, 0)
        green_time = 13
    elif 10 <= vehicle_count <= 16:
        density = "MEDIUM"
        color = (0, 255, 255)
        green_time = 18
    else:
        density = "HEAVY"
        color = (0, 0, 255)
        green_time = 60

    # Resize frame
    resized_frame = cv2.resize(frame, (1100, 650))

    # Smaller panel size
    panel_width = 380
    panel_height = 130

    x_start = 1100 - panel_width - 20
    y_start = 650 - panel_height - 20

    overlay = resized_frame.copy()

    # Draw panel
    cv2.rectangle(overlay,
                (x_start, y_start),
                (x_start + panel_width, y_start + panel_height),
                (0, 0, 0), -1)

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, resized_frame, 1 - alpha, 0, resized_frame)

    # Smaller clean text
    cv2.putText(resized_frame, f"Vehicles: {vehicle_count}",
                (x_start + 20, y_start + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(resized_frame, f"Density: {density}",
                (x_start + 20, y_start + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(resized_frame, f"Estimated Green Time: {green_time} sec",
                (x_start + 20, y_start + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # Create overlay for panel
    overlay = resized_frame.copy()

    # Draw semi-transparent rectangle (top-left panel)
    cv2.rectangle(overlay, (10, 10), (450, 180), (0, 0, 0), -1)


    # Center window
    screen_width = 1920
    screen_height = 1080
    x_pos = (screen_width - 1100) // 2
    y_pos = (screen_height - 650) // 2
    cv2.moveWindow("Traffic Detection", x_pos, y_pos)

    cv2.imshow("Traffic Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()