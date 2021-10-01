import numpy as np
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
chess_board_size = (7, 5)
frame_size = (1920, 1080)
objp = np.zeros((chess_board_size[0]*chess_board_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chess_board_size[0],0:chess_board_size[1]].T.reshape(-1,2)
objp *= 0.025

cap = cv.VideoCapture(0)
# Same dimensions as calibrated
cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_size[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_size[1])
capturing_images = True
no_captured_images = 0
while capturing_images:
    ret, frame = cap.read()    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #-- remember, OpenCV stores color images in Blue,
    ret, corners = cv.findChessboardCorners(gray, chess_board_size, None)
    # if ids is not None and ids[0] == id_to_find:
    if ret:        
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)        
        # Draw and display the corners
        frame_drawn = frame.copy()
        cv.drawChessboardCorners(frame_drawn, chess_board_size, corners2, ret)    
        cv.imshow('frame', frame_drawn)
    else:
        cv.imshow('frame', frame)

    key = cv.waitKey(1) & 0xFF
    # Save to file:
    if key == 13:
        # capturing_images = False 
        print("Saving picture")       
        cv.imwrite("./calibration_images/"+str(no_captured_images)+".png", frame)
        no_captured_images+=1
    # Load from file:
    elif key==27:
        capturing_images = False
        print("Exiting image capture")       
        
# print(objp)
cap.release()
cv.destroyAllWindows()
# quit()
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('./calibration_images/*.png')
not_found = 0
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, chess_board_size, None)
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, chess_board_size, corners2, ret)
        cv.imshow('img', img)
        # key = cv.waitKey(1) & 0xFF        
        key = cv.waitKey(5000) & 0xFF        
    else:
        print("Not found!")
        not_found += 1
cv.destroyAllWindows()
# quit()
print("Total not found:",not_found)
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frame_size, None, None)
# img = cv.imread('./calibration_images/12.png')
h,  w = img.shape[:2]

newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
print(newcameramtx)
np.savetxt('camera_params.txt', mtx, delimiter=',')
np.savetxt('dist_coefs.txt', dist, delimiter=',')
# undistort
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints))) 
