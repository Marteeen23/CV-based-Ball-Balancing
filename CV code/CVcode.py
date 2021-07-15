# Import Modules
import cv2
import time
import numpy as np

# Start capturing
vs = cv2.VideoCapture(0)

# Fps variable
fps = 0

# Start time
start = time.time()

# params that need tuning
dpi = 1.2
minDist = 500
param1 = 130
param2 = 30
minRadius = 0
maxRadius = 0

# Circle properties
Ccolor = (0, 0, 255)                                                                                                                                        # Red
Cthickness = 2

# Fps text properties
font = cv2.FONT_HERSHEY_SIMPLEX
position = (0, 25)                                                                                                                                          # Position of lower left corner of text, (0, 0) is top left of screen
FPScolor = (0, 255, 0)                                                                                                                                      # Green
FPSthickness = 2

while vs.isOpened() :
    # Read frame
    success, frame = vs.read()

    if not success : continue

    # Was trying gpu acceleration
    # gpuFrame = cv2.cuda_GpuMat()
    # gpuFrame.upload(frame)

    # Convert greyscale
    frameCircle = frame.copy()
    framegrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect circles in the image
    circles = cv2.HoughCircles(framegrey, cv2.HOUGH_GRADIENT, dpi, minDist, param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)

    if circles is not None :
        # Convert x, y, r to integers
        circles = np.round(circles[0, :]).astype("int")

        for (xCenter, yCenter, radius) in circles :
            cv2.circle(frameCircle, (xCenter, yCenter), radius, Ccolor, thickness)                                                                          # Make the circles

    frame = cv2.putText(frame, "FPS : " + str(int(fps)), position, font, 1, FPScolor, FPSthickness, cv2.LINE_AA)                                            # Display fps

    cv2.imshow("Real", frame)
    cv2.imshow("Circle", frameCircle)
    
    end = time.time()

    fps = 1 / (end - start)                                                                                                                                 # Fps Logic (I know a built in method exists)

    start = end

    # Press Escape key to stop
    if cv2.waitKey(2) & 0xFF == 27 :
        break

cv2.destroyAllWindows()

vs.release()
