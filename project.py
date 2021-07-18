# Import Modules
import cv2
import time
import numpy as np
# Start capturing
vs = cv2.VideoCapture('/home/joshuadsilva/Desktop/Python/vid1.mp4')
#frame=cv2.imread("/home/joshuadsilva/Desktop/Python/img1.jpeg")
# params that need tuning
dpi = 1.2
minDist = 500
param1 = 130
param2 = 20
minRadius = 0
maxRadius = 30
# Circle properties
Ccolor = (0, 0, 255)                                                                                                                                        # Red
Cthickness = 2
# Fps text properties
font = cv2.FONT_HERSHEY_SIMPLEX
position = (0, 25)      
position2 = (0, 50)                                                                                                                                     # Position of lower left corner of text, (0, 0) is top left of screen
FPScolor = (0, 255, 0)                                                                                                                                      # Green
FPSthickness = 2
# k=0
while True :
    # k+=1
    # Read frame
    success, frame = vs.read()
    now = time.time()
    frameLimit = 20.0
    frame=cv2.resize(frame, (int(frame.shape[1]*0.7), int(frame.shape[0]*0.7)), interpolation = cv2.INTER_AREA)
    if not success : continue
    # Convert greyscale
    frameCircle = frame.copy()
    framegrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_b = np.array([0, 0, 130])
    u_b = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, l_b, u_b)

    ret, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    # Detect circles in the image
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, dpi, minDist, param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)
    if circles is not None :
        # Convert x, y, r to integers
        circles = np.round(circles[0, :]).astype("int")

        for (xCenter, yCenter, radius) in circles :
            cv2.circle(frameCircle, (xCenter, yCenter), radius, Ccolor, Cthickness)                                                                          # Make the circles
            cv2.circle(frameCircle, (xCenter, yCenter), radius, Ccolor, Cthickness)  
            frameCircle = cv2.putText(frameCircle, "centre: " + str(xCenter) + '  ' + str(yCenter), position, font, 1, FPScolor, FPSthickness, cv2.LINE_AA,)                                                                        # Make the circles

    #frame = cv2.putText(frameCircle, "FPS : " + str(int(fps)), position2, font, 1, FPScolor, FPSthickness, cv2.LINE_AA,)      

    cv2.imshow("circle",frameCircle)
    #cv2.imshow("circle",framegrey)
    #cv2.imshow("circle",thresh)
    key = cv2.waitKey(1)
    if key == 27:
        break

    timeDiff = time.time() - now
    if (timeDiff < 1.0/(frameLimit)): 
        time.sleep( 1.0/(frameLimit) - timeDiff )
# cv2.waitKey(0)
vs.release()
cv2.destroyAllWindows()