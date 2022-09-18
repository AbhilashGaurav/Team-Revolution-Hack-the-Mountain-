# All copyright and license information belongs to the team memebers of Team Revolution

import cv2

# Method 2
vidCap = cv2.VideoCapture("Vehicle_moving.mp4")#or type the video file name in place of 0

# initilize OpenCV - Background Subtractor for KNN and MOG2
BS_KNN = cv2.createBackgroundSubtractorKNN()
BS_MOG2 = cv2.createBackgroundSubtractorMOG2()

vehile = 0
validVehiles = []
while vidCap.isOpened():
    ret, frame = vidCap.read() # reads the next frame

    # extract the foreground mask
    fgMask = BS_MOG2.apply(frame)
    
    # draw the reference traffic lines
    cv2.line(frame, (220,450), (1000,450), (0,0,255), 2) # RED Line
    cv2.line(frame, (220,440), (1000,440), (0,255,0), 1) # GREEN Offset ABOVE
    cv2.line(frame, (220,460), (1000,460), (0,255,0), 1) # GREEN Offset BELOW
    
    # extract the contours
    conts, _ = cv2.findContours(fgMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in conts:
        x, y, w, h = cv2.boundingRect(c)
        
        # ignore the small contours in size
        visibleVehile = (w > 60) and (h > 60)
        if not visibleVehile:
            continue
        
        # remove the distraction on the road; consider only the objects on ROAD
        if x > 250 and x < 900 and y > 200:
            # draw the bounding rectangle for all contours
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            xMid = int((x + (x+w))/2)
            yMid = int((y + (y+h))/2)
            cv2.circle(frame, (xMid,yMid),5,(0,0,255),5)

            # add all valid vehiles into List Array
            validVehiles.append((xMid,yMid))
#             cv2.waitKey(0) # debugging purpose

            for (vX, vY) in validVehiles:
                if vY > 450 and vY < 456: # adjust this for the frame jumping
                    vehile += 1
                    validVehiles.remove((vX,vY))
                    
                    # debugging purpose
#                     cv2.putText(frame, 'Y : {}'.format(yMid), (x, y-20), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

    # show the thresh and original video
    cv2.imshow('Foreground Mask', fgMask)
    cv2.putText(frame, 'Total Vehiles : {}'.format(vehile), (450, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
    cv2.imshow('Original Video', frame)
    
    # wait for any key to be pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release video capture
cv2.destroyAllWindows()
vidCap.release()
