import numpy as np
import cv2
import math

# -----------------------------------------------------------
# Load Video
# -----------------------------------------------------------
# open video file (use VideoCapture, not imread)
# vidCapture = cv2.VideoCapture('case.mp4')
# vidCapture = cv2.VideoCapture('rubics.mp4')
# vidCapture = cv2.VideoCapture('walk.mp4')
# vidCapture = cv2.VideoCapture('walksit.mp4')
# vidCapture = cv2.VideoCapture('test_person.mp4')
# vidCapture = cv2.VideoCapture('chainsnatch.mp4')
# vidCapture = cv2.VideoCapture('raghav.mp4')
# vidCapture = cv2.VideoCapture('tiger.mp4')
# vidCapture = cv2.VideoCapture('attacktiger.mp4')
vidCapture = cv2.VideoCapture('lioness.mp4')

ret, frame = vidCapture.read()
if not ret:
    raise ValueError("Could not read video")

# -----------------------------------------------------------
# Select ROI
# -----------------------------------------------------------
x, y, w, h = cv2.selectROI("Select Object", frame, fromCenter=False)
cv2.destroyWindow("Select Object")

track_window = (x, y, w, h)

# Reset video
vidCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)

# -----------------------------------------------------------
# Build Target Histogram (Hue channel only)
# -----------------------------------------------------------
roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hist_target = cv2.calcHist([hsv_roi], [0], None, [50], [0, 180]).astype(np.float32)
cv2.normalize(hist_target, hist_target, 0, 1, cv2.NORM_MINMAX)

# compute initial m00 for scaling normalization
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
initial_prob = cv2.calcBackProject([hsv_frame], [0], hist_target, [0,180], 1).astype(np.float32)
m00_0 = np.sum(initial_prob[y:y+h, x:x+w])

r1 = w / math.sqrt(m00_0)
r2 = h / math.sqrt(m00_0)

# -----------------------------------------------------------
# Tracking Loop
# -----------------------------------------------------------
max_iters = 100

while True:
    ret, frame = vidCapture.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    prob_map = cv2.calcBackProject([hsv], [0], hist_target, [0,180], 1).astype(np.float32)

    # -----------------------------------------------------------
    # Mean Shift Iteration
    # -----------------------------------------------------------
    for _ in range(max_iters):
        x, y, w, h = track_window

        window = prob_map[y:y+h, x:x+w]
        m00 = np.sum(window)

        if m00 < 1e-5:
            break

        # first moments
        xs = np.arange(w)[None, :]
        ys = np.arange(h)[:, None]

        m10 = np.sum(xs * window)      # x-axis weighted
        m01 = np.sum(ys * window)      # y-axis weighted

        xc = m10 / m00
        yc = m01 / m00

        new_x = int(x + xc - w/2)
        new_y = int(y + yc - h/2)

        H, W = frame.shape[:2]
        new_x = max(0, min(new_x, W - w))
        new_y = max(0, min(new_y, H - h))

        if abs(new_x - x) < 1 and abs(new_y - y) < 1:
            break

        track_window = (new_x, new_y, w, h)

    # -----------------------------------------------------------
    # Size Update (CamShift Width/Height)
    # -----------------------------------------------------------
    new_w = int(r1 * math.sqrt(m00))
    new_h = int(r2 * math.sqrt(m00))

    MIN_SIZE = 50
    MAX_SIZE_W = W // 2
    MAX_SIZE_H = H // 2

    new_w = min(max(new_w, MIN_SIZE), MAX_SIZE_W)
    new_h = min(max(new_h, MIN_SIZE), MAX_SIZE_H)

    center_x = x + xc
    center_y = y + yc

    new_x_tl = int(center_x - new_w / 2)
    new_y_tl = int(center_y - new_h / 2)

    new_x_tl = max(0, new_x_tl)
    new_y_tl = max(0, new_y_tl)

    if new_x_tl + new_w > W:
        new_w = W - new_x_tl
    if new_y_tl + new_h > H:
        new_h = H - new_y_tl

    track_window = (new_x_tl, new_y_tl, new_w, new_h)

    # -----------------------------------------------------------
    # Angle Calculation via Second-Order Moments
    # -----------------------------------------------------------
    x, y, w, h = track_window
    window = prob_map[y:y+h, x:x+w]

    m00 = np.sum(window)
    if m00 < 1e-5:
        continue

    xs = np.arange(w)[None, :]
    ys = np.arange(h)[:, None]

    m10 = np.sum(xs * window)
    m01 = np.sum(ys * window)

    xc = m10 / m00
    yc = m01 / m00

    # second order moments
    m20 = np.sum((xs**2) * window)
    m02 = np.sum((ys**2) * window)
    m11 = np.sum((xs * ys) * window)

    # centered moments
    mu20 = (m20 / m00) - xc**2
    mu02 = (m02 / m00) - yc**2
    mu11 = (m11 / m00) - xc * yc

    # orientation
    theta = 0.5 * math.atan2(2 * mu11, (mu20 - mu02))
    angle_deg = np.degrees(theta)

    # ellipse axis lengths
    A = mu20 + mu02
    B = math.sqrt(4 * mu11**2 + (mu20 - mu02)**2)

    lambda1 = (A + B) / 2
    lambda2 = (A - B) / 2

    major = 4 * math.sqrt(abs(lambda1))
    minor = 4 * math.sqrt(abs(lambda2))

    # -----------------------------------------------------------
    # Drawing Rotated Box
    # -----------------------------------------------------------
    center = (int(x + xc), int(y + yc))
    size = (int(max(major, 10)), int(max(minor, 10)))
    rect = (center, size, angle_deg)

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    cv2.polylines(frame, [box], True, (0, 0, 255), 2)

    # normal rect for reference
    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 1)

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

vidCapture.release()
cv2.destroyAllWindows()
cv2.waitKey(1)