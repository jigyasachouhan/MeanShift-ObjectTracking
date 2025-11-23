import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import cv2
import math
import sys

print("Options for CLI are: case, rubics, walk, walksit, test_person, chainsnatch, raghav, tiger, attacktiger, lioness")

name = sys.argv[1]

print("Doing it for:", name)

vidCapture = cv2.VideoCapture(f'videos/{name}.mp4')


ret, frame = vidCapture.read()
if not ret:
    raise ValueError("Could not read video")

# Let user select ROI
x, y, w, h = cv2.selectROI("Select Object", frame, fromCenter=False)
cv2.destroyWindow("Select Object")

# Reset video to start
vidCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)

track_window = (x, y, w, h)


roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hist_target = cv2.calcHist([hsv_roi], [0], None, [50], [0, 180]).astype(np.float32)
cv2.normalize(hist_target, hist_target, 0, 1, cv2.NORM_MINMAX)

hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
initial_prob_map = cv2.calcBackProject([hsv_frame], [0], hist_target, [0,180], 1).astype(np.float32)

# print(initial_prob_map)
m00_0 = np.sum(initial_prob_map[y:y+h, x:x+w])
print("m00_0:", m00_0) #Just some debugging code

r1 = w / math.sqrt(m00_0)
r2 = h / math.sqrt(m00_0)
print(m00_0)


max_iters = 100

while True:
    ret, frame = vidCapture.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    prob_map = cv2.calcBackProject([hsv], [0], hist_target, [0,180], 1)
    prob_map = prob_map.astype(np.float32)   # DO NOT NORMALIZE TO 0–1

    for _ in range(max_iters):
        x, y, w, h = track_window

        window = prob_map[y:y+h, x:x+w]

        m00 = np.sum(window)
        
        print("New m00:", m00)
        if m00 < 1e-3:
            break

        # moments in window coordinates
        m01 = np.sum(np.arange(w)[None, :] * window)    # x-weighted
        m10 = np.sum(np.arange(h)[:, None] * window)    # y-weighted

        xc = m01 / m00
        yc = m10 / m00


        # correct centroid → global frame
        new_x = int(x + xc - w/2)
        new_y = int(y + yc - h/2)

        height, width = frame.shape[:2]
        new_x = max(0, min(new_x, width - w))
        new_y = max(0, min(new_y, height - h))

        if abs(new_x - x) < 1 and abs(new_y - y) < 1:
            print("Converged")
            break

        track_window = (new_x, new_y, w, h)
    

    new_w = int(r1 * math.sqrt(m00))
    new_h = int(r2 * math.sqrt(m00))

    H, W = frame.shape[:2]

    # What this does is helps us to have sane values - and prevents collapse
    min = 12      
    maxw = W//2   
    maxh = H//2

    # Apply min/max limits to the size first
    new_w = min(max(new_w, min), maxw)
    new_h = min(max(new_h, min), maxh)

    center_x = x + xc
    center_y = y + yc

    new_x_tl = int(center_x - new_w / 2)
    new_y_tl = int(center_y - new_h / 2)

    new_x_tl = max(0, new_x_tl)
    new_y_tl = max(0, new_y_tl)

    #This is just to ensure we don't go out of bounds
    if new_x_tl + new_w > W:
        new_w = W - new_x_tl
    if new_y_tl + new_h > H:
        new_h = H - new_y_tl

    # 5. Final Update of the tracking window
    track_window = (new_x_tl, new_y_tl, new_w, new_h)

    x, y, w, h = track_window #The final values after clamping
    result = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Tracking', result)

    if cv2.waitKey(30) & 0xFF == 27:
        break

vidCapture.release() 
cv2.destroyAllWindows() 
cv2.waitKey(1)

    