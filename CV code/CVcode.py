# Import Modules
import cv2
import time
import numpy as np
import math

def getkernel (radius = 10) :
    Kernel = [[0 for i in range(2 * radius + 10)] for i in range(2*radius + 10)]

    Cx, Cy = radius + 5, radius + 5

    # radius = 10

    maxPos, maxNeg = 0.77065, 5                                                           # 1.5

    for x in range(2 * radius + 10) :
        for y in range(2 * radius + 10) :
            i = x - Cx
            j = y - Cy
            if (i * i + j * j) > radius * radius :
                Kernel[x][y] = - maxNeg * math.exp(-math.pow((i * i + j * j - radius * radius) / radius / radius, 0.5))
            else :
                Kernel[x][y] = maxPos * math.exp(-math.pow(abs(i * i + j * j - radius * radius) / radius / radius, 5))

    Kernel = np.array(Kernel)

    return Kernel

# 201
# Kernel = getkernel(9) / 120
Kernel = getkernel(23) / 10000

# Start capturing
# vs = cv2.VideoCapture(r'C:\Users\ARYAN SATPATHY\Downloads\WhatsApp Video 2021-07-17 at 12.08.55.mp4')
# vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture(r'C:\Users\ARYAN SATPATHY\Downloads\WhatsApp Video 2021-07-27 at 12.16.28.mp4')

fpsCap = 30

mode = 0

# Fps variable
fps = 0

# Start time
start = time.time()

# params that need tuning
dpi = 1.2
minDist = 500
param1 = 300
param2 = 30
minRadius = 0
maxRadius = 0

# Circle properties
Ccolor = (0, 0, 255)                                                                                                                                        # Red
Cthickness = 10

# Fps text properties
font = cv2.FONT_HERSHEY_SIMPLEX
position = (0, 25)                                                                                                                                          # Position of lower left corner of text, (0, 0) is top left of screen
FPScolor = (0, 255, 0)                                                                                                                                      # Green
FPSthickness = 2

'''
htmin, stmin, vtmin = 10, 20, 24
htmax, stmax, vtmax = 104, 153, 152
'''

# htmin, stmin, vtmin = 20, 10, 5

# htmax, stmax, vtmax = 35, 95, 150

htmin, stmin, vtmin = 2, 75, 176

htmax, stmax, vtmax = 20, 206, 251

minArea, maxArea = 600, 2000

def Hmin(val) :
    global htmin
    htmin = val
def Smin(val) :
    global stmin
    stmin = val
def Vmin(val) :
    global vtmin
    vtmin = val
def Hmax(val) :
    global htmax
    htmax = val
def Smax(val) :
    global stmax
    stmax = val
def Vmax(val) :
    global vtmax
    vtmax = val

'''
success, frame = vs.read()
cv2.imshow("Real", frame)
'''

'''
cv2.createTrackbar('Hmin', 'Real' , 0, 179, Hmin)
cv2.createTrackbar('Smin', 'Real' , 0, 255, Smin)
cv2.createTrackbar('Vmin', 'Real' , 0, 255, Vmin)
cv2.createTrackbar('Hmax', 'Real' , 0, 179, Hmax)
cv2.createTrackbar('Smax', 'Real' , 0, 255, Smax)
cv2.createTrackbar('Vmax', 'Real' , 0, 255, Vmax)
'''

try :
    i = 0
    while vs.isOpened() :
        # Read frame
        success, frame = vs.read()

        frameCircle = frame.copy()
        frameKernel = frame.copy()

        if not success : continue

        # Was trying gpu acceleration
        # gpuFrame = cv2.cuda_GpuMat()
        # gpuFrame.upload(frame)

        # Mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (htmin, stmin, vtmin), (htmax, stmax, vtmax))

        cv2.imshow("mask", mask)
        
        # Blur
        # HSV thresholding

        # Just testing out a kernel
        '''
        _mask = cv2.filter2D(mask, -1, Kernel)

        __mask = cv2.inRange(_mask, 230, 255)
        _mask = _mask.astype(np.uint8)
        cv2.imshow("Advanced Mask", _mask)

        _mask = cv2.bitwise_and(_mask, _mask, mask = __mask)

        cv2.CHAIN_APPROX_SIMPLE, look up its alternatives

        Number of edges in contour
        
        '''
        _mask = cv2.filter2D(mask, -1, Kernel)

        ind = np.unravel_index(np.argmax(_mask, axis=None), _mask.shape)
        y, x = ind

        print(_mask[ind])


        # if _mask[ind] > 100 :     # 170
        # if _mask[ind] > 3 :
        if _mask[ind] > 7 :
            cv2.circle(frameKernel, (x, y), 22, (0, 0, 255), 2)
        else :
            cv2.putText(frameKernel, "No Circle", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        
        # _mask = _mask * 255 / _mask.argmax()
        cv2.imshow("Kernel Circle", frameKernel)
        

        # Detect Circles :
        if mode == 0 :
            # Contours
            contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) :
                _contours = []
                for c in contours :
                    if minArea < cv2.contourArea(c) < maxArea :
                        _contours.append(c)

                if len(_contours) :
                    '''
                    Contour = max(_contours, key = cv2.contourArea)

                    ((x, y), radius) = cv2.minEnclosingCircle(Contour)

                    cv2.circle(frameCircle, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                    '''
                    # Remove this
                    # cv2.putText(frameCircle, '{}'.format(len(Contour)), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                    
                    __contours = []
                    for c in contours :
                        # if 30 > len(c) > 20 :
                        __contours.append(c)

                    if len(__contours) :
                        Contour = max(__contours, key = cv2.contourArea)

                        ((x, y), radius) = cv2.minEnclosingCircle(Contour)

                        # cv2.circle(frameCircle, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                        cv2.circle(frameCircle, (int(x), int(y)), 22, (0, 0, 255), 2)
                        cv2.drawContours(frame, __contours, -1, (0, 255, 0), thickness = 2)
                        cv2.putText(frameCircle, 'Radius : {}'.format(int(radius)), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness = 1)
                    
        
        if mode == 1 :
            # Hough Circle Detection
            circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dpi, minDist, param1 = param1, param2 = param2, minRadius = minRadius, maxRadius = maxRadius)

            if circles is not None :
                # Convert x, y, r to integers
                circles = np.round(circles[0, :]).astype("int")

                for (xCenter, yCenter, radius) in circles :
                    cv2.circle(frameCircle, (xCenter, yCenter), radius, Ccolor, Cthickness)                                                                          # Make the circles
        

        frame = cv2.putText(frame, "FPS : " + str(int(fps)), position, font, 1, FPScolor, FPSthickness, cv2.LINE_AA)                                            # Display fps

        cv2.imshow("Real", frame)
        cv2.imshow("Circle", frameCircle)
        
        end = time.time()

        if (end - start) < 1 / fpsCap :
            time.sleep(1 / fpsCap - (end - start))

        fps = fpsCap                                                                                                                                 # Fps Logic (I know a built in method exists)

        start = end

        if i == 0 :
            while True :
                key = cv2.waitKey(2)
                if key & 0xFF == 32 :
                    break
            i += 1

        # Press Escape key to stop
        key = cv2.waitKey(2)
        if key & 0xFF == 27 :
            break
        # Press space for pause of 5 seconds
        if key & 0xFF == 32 :
            print((x, y, radius))
            while (end < start + 5) :
                end = time.time()
except :
    print(htmin, stmin, vtmin, htmax, stmax, vtmax)

cv2.destroyAllWindows()

vs.release()
