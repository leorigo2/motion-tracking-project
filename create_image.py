import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

data = pd.read_csv('mse_per_id.csv', sep=',', header=None, names=["Object ID", "mse"])  

colors = ['green' if mse < 1 else 'red' for mse in data["mse"]]

outliers = [obj_id if mse >= 1 else None for obj_id, mse in zip(data["Object ID"], data["mse"])]

def load_outliers(filename):
    history = {}
    with open(filename, 'r') as f:
        obj_id = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Object ID"):
                obj_id = int(line.split()[2][:-1])
                if obj_id in outliers:
                    history[obj_id] = []
            elif obj_id in outliers:
                x_str, y_str = line.split(',')
                x = float(x_str.strip())
                y = float(y_str.strip())
                history[obj_id].append((int(x), int(y)))

    return history

def draw_trajectories(history, image, color, dot_radius=1):

    for obj_id, points in history.items():
        for i in range(1, len(points)):
            pt1 = points[i - 1]
            pt2 = points[i]
            cv2.line(image, pt1, pt2, color, 2)
            cv2.circle(image, pt2, dot_radius, color, -1)
        cv2.putText(image, str(obj_id), (pt2[0] + 5, pt2[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


plt.figure(figsize=(14, 6))
plt.bar(data["Object ID"].astype(str), data["mse"], color=colors, edgecolor='black')
plt.xticks(rotation=90, fontsize=8)
plt.xlabel("Object ID")
plt.ylabel("MSE Value")
plt.title("Bar Chart of MSE per Object ID")
plt.grid(axis='y', linestyle='--', alpha=0.8)
plt.savefig('mse_per_object_id.png')
plt.show()



motion_outliers = load_outliers('motion_history.txt')
pred_outliers = load_outliers('tracked_history.txt')

image = np.zeros((600, 800, 3), dtype=np.uint8)
image = draw_trajectories(motion_outliers, image, (0, 255, 0))
image = draw_trajectories(pred_outliers, image, (255, 0, 0))
cv2.imshow('Trajectories', image)
cv2.waitKey(0)
cv2.imwrite('outliers_trajectories.png', image)
cv2.destroyAllWindows()




