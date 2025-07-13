from collections import defaultdict
import cv2

def get_side(x, y, frame_w, frame_h, margin=0.2):
    if x < margin * frame_w:
        return "left"
    elif x > (1 - margin) * frame_w:
        return "right"
    elif y < margin * frame_h:
        return "top"
    elif y > (1 - margin) * frame_h:
        return "bottom"
    else:
        return "center"
    
def extrapolate_side(p1, p2, frame_w, frame_h, length=1000):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    new_x = p2[0] + dx * length
    new_y = p2[1] + dy * length
    return get_side(new_x, new_y, frame_w, frame_h)


object_positions = defaultdict(list)

image = cv2.imread("material/video0/reference.jpeg")

with open("material/video0/annotations.txt", "r") as f:
    for line in f:
        parts = line.strip().split()

        x1, y1, x2, y2 = map(int, parts[1:5])
        obj_id = int(parts[0])
        cx, cy = (x1, y1)
        
        object_positions[obj_id].append((cx, cy))

frequency_counter = defaultdict(int)
frame_w, frame_h = image.shape[1], image.shape[0]
for obj_id, positions in object_positions.items():
    first = positions[0]
    last = positions[-1]
    
    entry = get_side(first[0], first[1], frame_w, frame_h)
    exit = get_side(last[0], last[1], frame_w, frame_h)

    if entry == "center" or exit == entry:
        entry = extrapolate_side(positions[1], positions[0], frame_w, frame_h)
    
    if exit == "center" or exit == entry:
        exit = extrapolate_side(positions[-2], positions[-1], frame_w, frame_h)
    
    if entry != "center" and exit != "center":
        frequency_counter[(entry, exit)] += 1


sorted_counter = sorted(frequency_counter.items(), key=lambda x: x[1], reverse=True)
print(sorted_counter)