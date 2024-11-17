from tracker import Tracker
from ultralytics import YOLO
from metrics import generate_color,poly_containment_ratio,calculate_distance, is_wrong_route
import numpy as np
import configparser
import time
import ast
import cv2
from shapely.geometry import Polygon, box
import os

#Configs
config = configparser.ConfigParser()
config.read("./config/config.ini")
show_gui = config['DEFAULT'].getboolean('show_gui')
print("SHOW GUI:",show_gui)
source = config['DEFAULT'].get('source')
print("Source File:",source)
points = config['DEFAULT'].get('ROI')
road_corners = ast.literal_eval(f"[{points}]")
prev_distances = {}
polygon = Polygon(road_corners)
points_array = np.array(road_corners, dtype=np.int32)
vehicle_classes = [2, 3, 5, 6, 7]
time_gap = config['DEFAULT'].getint('time')
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)
tolerance = 10

model = YOLO('./models/best.pt')

cap = cv2.VideoCapture(source)

# Reference point on the road
reference_point = (18, 604)
last_saved_time = -1
# Tracker
tracker = Tracker()

frame_count = 0

min_x = min(point[0] for point in points_array)
max_x = max(point[0] for point in points_array)
min_y = min(point[1] for point in points_array)
max_y = max(point[1] for point in points_array)

while cap.isOpened():
    print("reading frame",frame_count)
    ret, frame = cap.read()
    
    if not ret:
        break
    frame = cv2.resize(frame, (1080, 720))
    frame_count+=1
    # if (frame_count % 2)!=0:
    #     continue
    results = model.predict(frame)

    detections = []

    for result in results[0].boxes:
            print(result.xyxy)
            x1,y1,x2,y2 = result.xyxy[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            score = float(result.conf)
            detections.append([x1, y1, x2, y2, score])
                
    detections = np.array(detections)

    tracker_outputs = tracker.update(frame,detections)

    for track in tracker.tracks:
        bbox = track.bbox
        track_id = track.track_id
        track_id = int(track_id)
        x1, y1, x2, y2 = bbox
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        inside = False
        if (min_x - tolerance <= x1 <= max_x + tolerance) and (min_x - tolerance <= x2 <= max_x + tolerance) and \
        (min_y - tolerance <= y1 <= max_y + tolerance) and (min_y - tolerance <= y2 <= max_y + tolerance):
            inside = True 
        
        color = (0, 0, 255)  
        status = "Smoke"  

        if inside:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{track_id}: {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            current_time = time.time()
            if last_saved_time == -1 or current_time - last_saved_time > time_gap:  # time_gap in seconds
                # Save detection result
                timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime(current_time))
                result_file_path = f"./results/{timestamp}.png"  # Adjust path as needed
      
                cv2.imwrite(result_file_path, frame)
                last_saved_time = current_time
    
    cv2.polylines(frame, [points_array], isClosed=True, color=(0, 255, 0), thickness=2)
    frame = cv2.resize(frame,(540,540))
    if(show_gui):
        cv2.imshow("Smoke detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII code for the escape key
        break
print("Releasing video")
cap.release()
cv2.destroyAllWindows()