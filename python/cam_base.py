import cv2
import cv2.cv as cv
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv.CV_CAP_PROP_FPS, 1);
cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 320)  
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 240) 
cv2.namedWindow("camera_Capture", cv.CV_WINDOW_AUTOSIZE)  
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("camera_Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 
