import cv2 as cv
import numpy as np
import glob
from matplotlib import pyplot as plt

chessboardSize = (11, 7)
frameSize = (1024, 576)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0]*chessboardSize[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) * 30

print(objp)

 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.
 
leftImages = glob.glob('dataset/data/imgs/leftcamera/*.png')
rightImages = glob.glob('dataset/data/imgs/rightcamera/*.png')

for imgLeft, imgRight in zip(leftImages, rightImages):
    imgL = cv.imread(imgLeft)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
 
    imgR = cv.imread(imgRight)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR:
        objpoints.append(objp)
 
        corners2L = cv.cornerSubPix(grayL,cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(corners2L)

        corners2R = cv.cornerSubPix(grayR,cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(corners2R)
 
        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, corners2L, retL)
        cv.imshow('Left img', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, corners2R, retR)
        cv.imshow('Right img', imgR)
        cv.waitKey(100)
 
cv.destroyAllWindows()


###########  CALIBRATION #################################


retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
hL, wL = imgL.shape[:2]
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(mtxL, distL, (wL, hL), 1, (wL, hL))

retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)
hR, wR = imgR.shape[:2]
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(mtxR, distR, (wR, hR), 1, (wR, hR))


################  STEREO VISION CALIBRATION ##########################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# here we fix the intrinsic camera matrixes so that only rot, trns, emat and fmat are calculated
# hence intrinsic parameters are the same

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# this step is performed to transformation between the two cameras and calculate Essential and fundamental matrixes
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, (wL, hL), criteria = criteria, flags=flags) 


###############  STEREO RECTIFICATION  ##############################

rectrifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, 
                                                                           (wL, hL), rot, trans)

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("saving parameters!")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write("stereoMapL_x", stereoMapL[0])
cv_file.write("stereoMapL_y", stereoMapL[1])
cv_file.write("stereoMapR_x", stereoMapR[0])
cv_file.write("stereoMapR_y", stereoMapR[1])

cv_file.release()