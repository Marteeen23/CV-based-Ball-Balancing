# Import Modules
import cv2
import time
import numpy as np
import math
from serial import Serial

def getSkernel (edge = 10) :
    offset = [5, 10, 15]
    if edge <= 10 : offset = offset[0]
    elif edge <= 30 : offset = offset[1]
    else : offset = offset[2]
    
    Kernel = [[0 for i in range(edge + offset)] for i in range(edge + offset)]

    Cx, Cy = math.ceil((edge + offset) / 2), math.ceil((edge + offset) / 2)

    # radius = 10

    maxPos, maxNeg = 0.77065, 5                                                           # 1.5

    for x in range(edge + offset) :
        for y in range(edge + offset) :
            i = x - Cx
            j = y - Cy
            if abs(i) > edge // 2 or abs(j) > edge // 2 :
                Kernel[x][y] = - maxNeg * math.exp(-math.pow((max(abs(i), abs(j)) ** 2 - (edge // 2) ** 2) / ((edge // 2) ** 2), 0.5))   # 0.5
            else :
                Kernel[x][y] = maxPos * math.exp(-abs(math.pow((min(abs(i), abs(j)) ** 2 - (edge // 2) ** 2) / ((edge // 2) ** 2), 5)))

    Kernel = np.array(Kernel)

    return Kernel

def getCkernel (radius = 10) :
    offset = [5, 10, 15]
    if radius <= 10 : offset = offset[0]
    elif radius <= 30 : offset = offset[1]
    else : offset = offset[2]
    
    Kernel = [[0 for i in range(2 * radius + offset)] for i in range(2*radius + offset)]

    Cx, Cy = radius + math.ceil(offset / 2), radius + math.ceil(offset / 2)

    # radius = 10

    maxPos, maxNeg = 0.77065, 5                                                           # 1.5

    for x in range(2 * radius + offset) :
        for y in range(2 * radius + offset) :
            i = x - Cx
            j = y - Cy
            if (i * i + j * j) > radius * radius :
                Kernel[x][y] = - maxNeg * math.exp(-math.pow((i * i + j * j - radius * radius) / radius / radius, 0.5))   # 0.5
            else :
                Kernel[x][y] = maxPos * math.exp(-math.pow(abs(i * i + j * j - radius * radius) / radius / radius, 5))

    Kernel = np.array(Kernel)

    return Kernel

BallKernel = getCkernel(40) / 10000
CornerKernel = getSkernel(50) / 10000

def visualizeKernel(Kernel) :
    Max = Kernel.argmax()
    Max = Kernel[np.unravel_index(Max, Kernel.shape)]
    Min = Kernel.argmin()
    Min = Kernel[np.unravel_index(Min, Kernel.shape)]

    _min, _max = 0, 255

    slope = (_max - _min) / (Max - Min)

    Frame = Kernel.copy()

    Frame -= Min
    
    Frame *= slope

    Frame += _min

    Frame = Frame.astype(np.uint8)

    x, y = Kernel.shape

    Frame = cv2.resize(Frame, None, fx = int(500 / x), fy = int(500 / y), interpolation = cv2.INTER_LINEAR)

    cv2.imshow('Kernel Visualization', Frame)
    cv2.waitKey(0)

    cv2.destroyWindow('Kernel Visualization')

visualizeKernel(BallKernel)

# Serial Instance
try :
    arduino = Serial(port = 'COM4', baudrate = 115200, timeout = 0.1)
except :
    arduino = None

# Start capturing
vs = cv2.VideoCapture(r'C:\Users\ARYAN SATPATHY\Desktop\CV code\Video3.mp4')

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

# Corner properties
Cornercolor = (0, 255, 0)                                                                                                                                        # Green
Cornerradius = 3
Cornerthickness = -1

# Fps text properties
font = cv2.FONT_HERSHEY_SIMPLEX
position = (0, 25)                                                                                                                                          # Position of lower left corner of text, (0, 0) is top left of screen
FPScolor = (0, 255, 0)                                                                                                                                      # Green
FPSthickness = 2

Ballhtmin, Ballstmin, Ballvtmin = 6, 208, 141

Ballhtmax, Ballstmax, Ballvtmax = 46, 255, 255

Cornerhtmin, Cornerstmin, Cornervtmin = 27, 61, 33

Cornerhtmax, Cornerstmax, Cornervtmax = 47, 255, 255

BallminArea, BallmaxArea = 4000, 8000

BallminPoints, BallmaxPoints = 20, 30
    
i = 0
lastPos = (0, 0)
lastVel = (0, 0)
lastcntrs = [[0,0], [0,0], [0,0], [0,0]]

def updateLoop() :
    global vs
    global Ballhtmin, Ballstmin, Ballvtmin, Ballhtmax, Ballstmax, Ballvtmax
    global Cornerhtmin, Cornerstmin, Cornervtmin, Cornerhtmax, Cornerstmax, Cornervtmax
    global BallminArea, BallmaxArea, BallminPoints, BallmaxPoints
    global i, lastcntrs
    global font, position, FPScolor, FPSthickness
    global Cornercolor, Cornerradius, Cornerthickness
    global Ccolor, Cthickness
    global dpi, minDist, param1, param2, minRadius, maxRadius
    global start
    global fps
    global mode
    global fpsCap
    global arduino
    global BallKernel, CornerKernel
    
    success, frame = vs.read()

    frameCircle = frame.copy()
    frameKernel = frame.copy()

    if not success : return None

    # Was trying gpu acceleration
    # gpuFrame = cv2.cuda_GpuMat()
    # gpuFrame.upload(frame)

    # HSV and perspective
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    CornerMask = cv2.inRange(hsv, (Cornerhtmin, Cornerstmin, Cornervtmin), (Cornerhtmax, Cornerstmax, Cornervtmax))
    # _CornerMask = cv2.filter2D(CornerMask, -1, CornerKernel)
    # garbage, _CornerMask = cv2. threshold(_CornerMask, 200, 255, 0)
    _CornerMask = CornerMask
    cv2.imshow('Corners', _CornerMask)

    CornerContours, heirarchy = cv2. findContours(_CornerMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    NE, SE, SW, NW = [], [], [], []

    cv2.drawContours(frameCircle, CornerContours, -1, (0, 255, 255), thickness = 2)

    Rows, Columns, space = frame.shape

    cv2.line(frameCircle, (150, 0), (150, Rows), (0, 255, 255), thickness = 2)
    cv2.line(frameCircle, (0, 250), (Columns, 250), (0, 255, 255), thickness = 2)

    for c in CornerContours :
        M = cv2.moments(c)

        # calculate x,y coordinate of center
        cX = int(M["m10"] / (M["m00"]+0.00001))
        cY = int(M["m01"] / (M["m00"]+0.00001))

        if cX <= 150 and cY <= 250:
            NW.append(c)
        elif cX > 150 and cY <= 250 :
            NE.append(c)
        elif cX <= 150 and cY > 250 :
            SW.append(c)
        elif cX > 150 and cY > 250 :
            SE.append(c)

    if len(NE) * len(SE) * len(NW) * len(SW) != 0 :
        NE = max(NE, key = cv2.contourArea)
        SE = max(SE, key = cv2.contourArea)
        SW = max(SW, key = cv2.contourArea)
        NW = max(NW, key = cv2.contourArea)

        _ = 0
        # offset = [(-40, -40), (40, -40), (-40, 40), (40, 40)]
        offset = [(-0, -0), (0, -0), (-0, 0), (0, 0)]
        for c in [NW, NE, SW, SE] :
            M = cv2.moments(c)

            # calculate x,y coordinate of center
            cX = int(M["m10"] / (M["m00"]+0.00001))
            cY = int(M["m01"] / (M["m00"]+0.00001))
            lastcntrs[_]=[cX + offset[_][0], cY + offset[_][1]]
            cv2.circle(frameCircle, (cX, cY), Cornerradius, Cornercolor, Cornerthickness)
            cv2.putText(frameCircle, "centroid +"+ str(cX) + " " + str(cY), (cX, cY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            _ += 1
        
        pts1 = np.float32(lastcntrs)
        pts2 = np.float32([[0,0],[400,0],[0,400], [400,400]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        hsv = cv2.warpPerspective(hsv,M,(400, 400))
        frameKernel = cv2.warpPerspective(frameKernel,M,(400, 400))
        # frame = cv2.warpPerspective(frame,M,(400, 400))
        frameCircle = cv2.warpPerspective(frameCircle,M,(400, 400))
    else :
        pr = ''
        if len(NE) == 0 : pr += 'NE '
        if len(SE) == 0 : pr += 'SE '
        if len(NW) == 0 : pr += 'NW '
        if len(SW) == 0 : pr += 'SW '
        print(pr)

    
    # Mask
    mask = cv2.inRange(hsv, (Ballhtmin, Ballstmin, Ballvtmin), (Ballhtmax, Ballstmax, Ballvtmax))

    cv2.imshow("mask", mask)

    # Just testing out a kernel
    _mask = cv2.filter2D(mask, -1, BallKernel)

    ind = np.unravel_index(np.argmax(_mask, axis=None), _mask.shape)
    y, x = ind

    if _mask[ind] > 7 :
        cv2.circle(frameKernel, (x, y), 38, (0, 0, 255), 2)
    else :
        cv2.putText(frameKernel, "No Circle", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    
    cv2.imshow("Kernel Circle", frameKernel)

    # Detect Circles :
    if mode == 0 :
        # Contours
        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) :
            _contours = []
            for c in contours :
                if BallminArea < cv2.contourArea(c) < BallmaxArea :
                    _contours.append(c)

            if len(_contours) :
                # Remove this
                # cv2.putText(frameCircle, '{}'.format(len(Contour)), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

                
                __contours = []
                for c in contours :
                    # if BallmaxPoints > len(c) > BallminPoints :
                    __contours.append(c)

                if len(__contours) :
                    Contour = max(__contours, key = cv2.contourArea)

                    ((x, y), radius) = cv2.minEnclosingCircle(Contour)

                    cv2.circle(frameCircle, (int(x), int(y)), 38, (0, 0, 255), 2)
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
    

    frame = cv2.putText(frame, "FPS : " + str(int(fpsCap)), position, font, 1, FPScolor, FPSthickness, cv2.LINE_AA)                                            # Display fps

    cv2.imshow("Real", frame)
    cv2.imshow("Circle", frameCircle)

    if i == 0 :
        while True :
            key = cv2.waitKey(2)
            if key & 0xFF == 32 :
                break
        i += 1

    if arduino is not None :
        arduino.write(bytes('{} {}'.format(x, y), 'utf-8'))

    # Press Escape key to stop
    key = cv2.waitKey(2)
    if key & 0xFF == 27 :
        return 1
    # Press space for pause of 5 seconds
    if key & 0xFF == 32 :
        print((x, y, radius))
        return 2
        '''
        while (end < start + 5) :
            end = time.time()
        '''
    # Press s for slow mo
    if key & 0xFF == 115 :
        fpsCap = fpsCap / 2

    if arduino is not None :
        print("Serial : ", arduino.readline().decode('utf-8').rstrip())

cooldown = 0
flag = True
while True :
    now = time.time()
    if (now - start) >= max(cooldown, 1 / fpsCap) :
        cooldown = 0
        flag = True
        start = now
    else :
        flag = False
    if flag :
        try :
            retCode = updateLoop()
            
            if retCode == 1 :
                break
            if retCode == 2 :
                cooldown = 5
        except :
            break

cv2.destroyAllWindows()

vs.release()
