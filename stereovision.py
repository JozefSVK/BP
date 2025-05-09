import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

# camera parameters to undistort and rectify images
cv_file = cv.FileStorage()
cv_file.open('stereoMap.xml', cv.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

leftImages = glob.glob('dataset/data/imgs/leftcamera/*.png')
rightImages = glob.glob('dataset/data/imgs/rightcamera/*.png')

for imgLeft, imgRight in zip(leftImages, rightImages):
    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)

    imgL = cv.remap(imgL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    imgR = cv.remap(imgR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

    plt.subplot(2, 1, 1)  # (rows, columns, index) 
    plt.imshow(imgL)
    plt.title('Left Image')
    plt.axis('off')

    plt.subplot(2, 1, 2)  # (rows, columns, index) 
    plt.imshow(imgR)
    plt.title('Right Image')
    plt.axis('off')

    plt.show()


