import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sort import Sort

tracker = Sort()
MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
KNN_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows = True)

THRESH = 120
MIN_AREA = 250

video = cv2.VideoCapture("material/video0/video.mp4")

width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video0.avi', fourcc, fps, (width, height))

subtractor = KNN_subtractor

def compute_mse_per_object(predicted_dict, actual_dict):
    mse_per_id = {}

    for object_id in predicted_dict:
        if object_id not in actual_dict:
            continue

        preds = np.array(predicted_dict[object_id])[:, 0]
        actuals = np.array(actual_dict[object_id])[:, 0]

        min_len = min(len(preds), len(actuals))
        if min_len >= 100:
            mse = mean_squared_error(actuals[:min_len], preds[:min_len])
            mse_per_id[object_id] = round(mse, 2)

    return mse_per_id

while True:
    ret, frame = video.read()

    if frame is None:
        break

    frame = cv2.GaussianBlur(frame, (25, 25), 0)

    diff = subtractor.apply(frame)

    _, motion_mask = cv2.threshold(diff.copy(), THRESH, 255, cv2.THRESH_BINARY)
    
    dilated_motion_mask = cv2.dilate(motion_mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),iterations = 2)

    contours, _ = cv2.findContours(dilated_motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 500: 
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append([x, y, x + w, y + h, 0.9]) 

    dets_array = np.array(detections) if detections else np.empty((0, 5))
    
    tracked_objects = tracker.update(dets_array)

    for d in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, d)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("Tracked", frame)
    cv2.imshow("Motion mask", dilated_motion_mask)

    out.write(frame)


    if cv2.waitKey(10) == ord("q"):
        break

video.release()
out.release()
cv2.destroyAllWindows()

motion = tracker.motion_trajectories
pred = tracker.tracked_trajectories

mse_per_id = compute_mse_per_object(pred, motion)

with open("mse_per_id.csv", "w") as f:
    for obj_id, mse in mse_per_id.items():
        f.write(f"{obj_id},{mse}\n")


with open("tracked_history.txt", "w") as f:
    for obj_id, centroids in motion.items():
        if len(centroids) < 30:
            continue
        f.write(f"Object ID {obj_id}:\n")
        for centroid in centroids:
            x, y = centroid
            f.write(f"{x[0]:.1f}, {y[0]:.1f}\n")
        f.write("\n")

with open("motion_history.txt", "w") as f:
    for obj_id, centroids in pred.items():
        if len(centroids) < 30:
            continue
        f.write(f"Object ID {obj_id}:\n")
        for centroid in centroids:
            x, y = centroid
            f.write(f"{x[0]:.1f}, {y[0]:.1f}\n")
        f.write("\n")


