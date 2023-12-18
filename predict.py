from ultralytics import YOLO
import cv2 as cv
import numpy as np

# Load a model
model = YOLO('./run/detect/train/weights/best.pt')  # load a custom model

# Define the polygonal zones
ash = (1080, 1920)
zone_1 = np.array([[0, ash[0] * 0.3], [0, ash[0]], [ash[1], ash[0]], [100, ash[0] * 0.3]], np.int32)
zone_1 = zone_1.reshape((-1, 1, 2))
zone_2 = np.array([[100, ash[0] * 0.3], [ash[1], ash[0]], [ash[1], ash[0] * 0.5], [ash[1] * 0.5, ash[0] * 0.3]], np.int32)
zone_2 = zone_2.reshape((-1, 1, 2))

cv.namedWindow('live', cv.WINDOW_NORMAL)
cap = cv.VideoCapture(r"C:\Users\soufi\OneDrive\Images\Pellicule\WIN_20231214_18_39_07_Pro.mp4")  # specify the path to your video file

while True:
    count_zone_1 = 0
    count_zone_2 = 0
    ret, img = cap.read()

    if not ret:
        break

    results = model(img)

    # Draw the polygonal zones on the image
    cv.polylines(img, [zone_1], isClosed=True, color=(255, 0, 0), thickness=2)
    cv.polylines(img, [zone_2], isClosed=True, color=(255, 255, 0), thickness=2)

    for result in results:
        # Accessing the boxes attribute
        boxes = result.boxes

        if boxes is not None:
            # Iterate through bounding boxes
            for box in boxes.xyxy:
                # Check if the lower points of the detected object are inside the polygonal zones
                cv.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                lower_point1_inside_zone_1 = cv.pointPolygonTest(zone_1, (int(box[0]), int(box[3])), False) >= 0
                lower_point2_inside_zone_1 = cv.pointPolygonTest(zone_1, (int(box[2]), int(box[3])), False) >= 0
                lower_point1_inside_zone_2 = cv.pointPolygonTest(zone_2, (int(box[0]), int(box[3])), False) >= 0
                lower_point2_inside_zone_2 = cv.pointPolygonTest(zone_2, (int(box[2]), int(box[3])), False) >= 0

                if lower_point1_inside_zone_1 and lower_point2_inside_zone_1:
                    count_zone_1 += 1
                    # cv.putText(img, f'Zone 1: {count_zone_1}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                    print("Person inside Zone 1!")

                if lower_point1_inside_zone_2 and lower_point2_inside_zone_2:
                    count_zone_2 += 1
                    # cv.putText(img, f'Zone 2: {count_zone_2}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
                    print("Person inside Zone 2!")
                cv.putText(img, f'Zone 1: {count_zone_1}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                cv.putText(img, f'Zone 2: {count_zone_2}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)


    cv.imshow('live', img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()