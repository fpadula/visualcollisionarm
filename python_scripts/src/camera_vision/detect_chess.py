import numpy as np
import cv2 as cv
import glob
import time

#--- Capture the videocamera (this may also be a video or a picture)
cap = cv.VideoCapture(0)
#-- Set the camera size as the one it was calibrated with
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
chess_board_size = (7, 5)
#-- Font for the text in the image
font = cv.FONT_HERSHEY_PLAIN
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
starting_img_number = 1
while True:

    #-- Read the camera frame
    ret, frame = cap.read()

    #-- Convert in gray scale
    gray    = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #-- remember, OpenCV stores color images in Blue, Green, Red

    ret, corners = cv.findChessboardCorners(gray, chess_board_size, None)
    # If found, add object points, image points (after refining them)
    if ret:        
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)        
        # Draw and display the corners
        cv.imwrite("./calibration_images/" + str(starting_img_number)+'.png', frame)     
        cv.drawChessboardCorners(frame, chess_board_size, corners2, ret)   
        starting_img_number+=1
        # time.sleep(5)

    #--- Display the frame
    cv.imshow('frame', frame)

    #--- use 'q' to quit
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break































