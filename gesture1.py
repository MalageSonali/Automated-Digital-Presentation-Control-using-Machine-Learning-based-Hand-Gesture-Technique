import cv2
import numpy as np
import math
import keyboard
import pyautogui as pg
import HandTrackingModule as htm
import time
import autopy


cap = cv2.VideoCapture(0)

     
while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[100:300, 100:300]
        
        
        cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        
         
    # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
     #extract skin colour image
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
   
        
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
    #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100) 
        
        
        
    #find contours
        contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
   #find contour of max area(hand)
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
    #approx the contour a little
        epsilon = 0.0005*cv2.arcLength(cnt,True)
        approx= cv2.approxPolyDP(cnt,epsilon,True)
       
        
    #make convex hull around hand
        hull = cv2.convexHull(cnt)
        
     #define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)
      
    #find the percentage of area not covered by hand in convex hull
        arearatio=((areahull-areacnt)/areacnt)*100
    
     #find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
    # l = no. of defects
        l=0
        
    #code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt= (100,180)
            
            
            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a+b+c)/2
            ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
            
            #distance between point and convex hull
            d=(2*ar)/a
            
            # apply cosine rule here
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            
        
            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d>30:
                l += 1
                cv2.circle(roi, far, 3, [255,0,0], -1)
            
            #draw lines around hand
            cv2.line(roi,start, end, [0,255,0], 2)
            
            
        l+=1
        
        #print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l==1:
            if areacnt<2000:
                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            else:
                if arearatio<12:
                    cv2.putText(frame,'Stop',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    pg.hotkey('escape')
                    k = cv2.waitKey(3000) & 0xFF

                elif arearatio<17.5:
                    cv2.putText(frame,'',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

                else:
                    #cv2.putText(frame,'',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    cv2.putText(frame,'Display Marker',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    ######################
                    wCam, hCam = 640, 480
                    frameR = 100     #Frame Reduction
                    smoothening = 7  #random value
                    ######################

                    pTime = 0
                    plocX, plocY = 0, 0
                    clocX, clocY = 0, 0
                    cap = cv2.VideoCapture(0)
                    cap.set(3, wCam)
                    cap.set(4, hCam)

                    detector = htm.handDetector(maxHands=1)
                    wScr, hScr = autopy.screen.size()

                    while True:
                        # Step1: Find the landmarks
                        keyboard.press_and_release('ctrl + l')
                        success, img = cap.read()
                        img = detector.findHands(img)
                        lmList, bbox = detector.findPosition(img)

                        
                        # Step2: Get the tip of the index and middle finger
                        if len(lmList) != 0:
                            x1, y1 = lmList[8][1:]
                            x2, y2 = lmList[12][1:]

                            # Step3: Check which fingers are up
                            fingers = detector.fingersUp()
                            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                                          (255, 0, 255), 2)

                            # Step4: Only Index Finger: Moving Mode
                            if fingers[1] == 1 and fingers[2] == 0:
                                #pygui.hotkey('ctrl', 'l')

                                # Step5: Convert the coordinates
                                x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
                                y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

                                # Step6: Smooth Values
                                clocX = plocX + (x3 - plocX) / smoothening
                                clocY = plocY + (y3 - plocY) / smoothening

                                # Step7: Move Mouse
                                autopy.mouse.move(wScr - clocX, clocY)
                                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                                plocX, plocY = clocX, clocY

                
                        # Step11: Frame rate
                        cTime = time.time()
                        fps = 1/(cTime-pTime)
                        pTime = cTime
                        cv2.putText(img, str(int(fps)), (28, 58), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)

                        # Step12: Display
                        cv2.imshow("Image", img)
                        cv2.waitKey(1)

            k = cv2.waitKey(3000) & 0xFF
                    
        elif l==2:
            cv2.putText(frame,'Presentation Mode',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            #k = cv2.waitKey(2000) & 0xFF
            if l==4:
                cv2.putText(frame,'previous slide',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                keyboard.press_and_release('left')
                k = cv2.waitKey(2000) & 0xFF

            elif l==5:
                cv2.putText(frame,'next slide',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                keyboard.press_and_release('right')
                k = cv2.waitKey(2000) & 0xFF

            #else:
                #break
                                             
            
        elif l==3:
         
              if arearatio<27:
                    cv2.putText(frame,'Video Mode',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    # keyboard.press_and_release('left')

                    if l==2:
                        cv2.putText(frame,'Play/Pause',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        pg.hotkey('alt', 'p')
                        k = cv2.waitKey(2000) & 0xFF


                    elif l==4:
                        cv2.putText(frame,'Volume Low',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        pg.hotkey('alt', 'down')
                        k = cv2.waitKey(2000) & 0xFF

                    elif l==5:
                        cv2.putText(frame,'Volume Up',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                        pg.hotkey('alt', 'up')
                        k = cv2.waitKey(2000) & 0xFF
            
                    
              else:
                    cv2.putText(frame,'',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
                    #break
                    
        elif l==4:
            cv2.putText(frame,'Display Marker',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            ######################
            wCam, hCam = 640, 480
            frameR = 100     #Frame Reduction
            smoothening = 7  #random value
            ######################

            pTime = 0
            plocX, plocY = 0, 0
            clocX, clocY = 0, 0
            cap = cv2.VideoCapture(0)
            cap.set(3, wCam)
            cap.set(4, hCam)

            detector = htm.handDetector(maxHands=1)
            wScr, hScr = autopy.screen.size()

            while True:
                # Step1: Find the landmarks
                keyboard.press_and_release('ctrl + l')
                success, img = cap.read()
                img = detector.findHands(img)
                lmList, bbox = detector.findPosition(img)

                
                # Step2: Get the tip of the index and middle finger
                if len(lmList) != 0:
                    x1, y1 = lmList[8][1:]
                    x2, y2 = lmList[12][1:]

                    # Step3: Check which fingers are up
                    fingers = detector.fingersUp()
                    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                                  (255, 0, 255), 2)

                    # Step4: Only Index Finger: Moving Mode
                    if fingers[1] == 1 and fingers[2] == 0:
                        #pygui.hotkey('ctrl', 'l')

                        # Step5: Convert the coordinates
                        x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
                        y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

                        # Step6: Smooth Values
                        clocX = plocX + (x3 - plocX) / smoothening
                        clocY = plocY + (y3 - plocY) / smoothening

                        # Step7: Move Mouse
                        autopy.mouse.move(wScr - clocX, clocY)
                        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        plocX, plocY = clocX, clocY

        
                # Step11: Frame rate
                cTime = time.time()
                fps = 1/(cTime-pTime)
                pTime = cTime
                cv2.putText(img, str(int(fps)), (28, 58), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)

                # Step12: Display
                cv2.imshow("Image", img)
                cv2.waitKey(1)

            k = cv2.waitKey(3000) & 0xFF

        elif l==5:
            cv2.putText(frame,'Start',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            pg.press('f5')
            k = cv2.waitKey(3000) & 0xFF
            
        elif l==6:
            cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            #pg.hotkey('alt', 'p')
            #k = cv2.waitKey(2000) & 0xFF
            
        else :
            cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
            
        #show the windows
        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)
    except:
        pass
        
    
    k = cv2.waitKey(3000) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    
    




