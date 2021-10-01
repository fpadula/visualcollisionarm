import numpy as np
import cv2 as cv
import glob

def draw(img, corners, imgpts):
    # imgpts = np.int32(imgpts).reshape(-1,2)
    # # draw ground floor in green
    # img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # # draw pillars in blue color
    # for i,j in zip(range(4),range(4,8)):
    #     img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # # draw top layer in red color
    # img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    # return img
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# Load previously saved data
mtx = np.loadtxt('camera_params.txt', delimiter=',')
dist = np.loadtxt('dist_coefs.txt', delimiter=',')
print(mtx)
print(dist)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((4*4,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:4].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
# axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
#                    [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

while True:
# for fname in glob.glob('*.jpg'):
    ret, frame = cap.read()
    # img = cv.imread(fname)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (4,4),None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        frame = draw(frame,corners2,imgpts)
    cv.imshow('img',frame)
    k = cv.waitKey(1) & 0xFF
    if k == ord('q'):
        cap.release()
        cv.destroyAllWindows()