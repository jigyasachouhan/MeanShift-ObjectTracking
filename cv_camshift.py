import cv2
import numpy as np
import sys

print("Options for CLI are: case, rubics, walk, walksit, test_person, chainsnatch, raghav, tiger, attacktiger, lioness")

name = sys.argv[1]

print("Doing it for:", name)

cap = cv2.VideoCapture(f'videos/{name}.mp4')

ret, frame = cap.read()

# Let user select ROI
roi = cv2.selectROI("Select Object", frame, False, False)
cv2.destroyWindow("Select Object")

x, y, w, h = roi
track_window = (x, y, w, h)

# Extract ROI for histogram
roi_frame = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

# This is important because low light values have broad hue values that can skew the histogram
mask = cv2.inRange(hsv_roi, np.array((0., 30., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# CamShift termination criteria: 10 iterations or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv2.CamShift(back_proj, track_window, term_crit)

    pts = cv2.boxPoints(ret)
    pts = pts.astype(int)  
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    cv2.imshow("CamShift Tracking", frame)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()

