import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import math
import time
import PySide6.QtCore
import igev_wrapper
import raft_wrapper

count = 1

def calculate_disparity(left_image, right_image, model):
    if(model == 'IGEV'):
        return igev_wrapper.run_igev_inference(left_image, right_image)
    elif(model == 'RAFT'):
        return raft_wrapper.run_raft_inference(left_image, right_image)

def func(left_image_path, right_image_path, donwscale=0.5, model='IGEV'):
    print('Starting')
    imgL = cv2.imread(left_image_path)
    imgR = cv2.imread(right_image_path)

    # downscale
    imgL = cv2.resize(imgL, None, fx=donwscale, fy=donwscale)
    imgR = cv2.resize(imgR, None, fx=donwscale, fy=donwscale)
    # print(imgL.shape[:2])

    # call wrapper
    disparityLR, disparityRL = calculate_disparity(imgL, imgR, model)

    # plt.imshow(disparityLR)
    # plt.show()
    # plt.imshow(disparityRL)
    # plt.show()

    imgL_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    imgR_rgb = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

    imgIR = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.uint8)
    imgIL = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.uint8)
    imgI = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.uint8)

    height, width = imgR.shape[:2]

    alpha = 0.5

    # start = time.time()
    # print("Starting")

    disparityIL = np.ones((imgL.shape[0], imgL.shape[1], 1), np.float32) * -1
    disparityIR = np.ones((imgL.shape[0], imgL.shape[1], 1), np.float32) * -1

    for y in range(height):
        for x in range(width):
            # if np.array_equal(imgIR[y, x], [0,0,0]):
            disparity = disparityLR[y, x]
            # if(disparity == 0 and (x+1 < width and x-1 > -1 and y+1 < height and y-1 > -1)):
            #     disparity = max(disparityLR[y,x+1], disparityLR[y, x-1], 
            #                     disparityLR[y+1,x], disparityLR[y-1, x])
            newX = math.floor(x - alpha*disparity)
            newX = max(0, min(newX, width-1))
            disparityIL[y,newX] = alpha*disparity
            # imgIL[y, newX] = imgL_rgb[y, x]

    # Fill holes
    for y in range(height):
        for x in range(1,width-1):
            if ((np.std([disparityIL[y,x-1], disparityIL[y,x], disparityIL[y,x+1]]) > 20 or disparityIL[y,x] == -1) and np.std([disparityIL[y,x-1], disparityIL[y,x+1]]) < 3):
                disparityIL[y,x] = np.average([disparityIL[y,x-1], disparityIL[y,x+1]])

    for y in range(height):
        for x in range(width):
            if (disparityIL[y, x] >= 0):
                imgIL[y, x] = imgL_rgb[y, x + int(np.round(disparityIL[y, x]))]


    for y in range(height):
        for x in range(width-1, -1, -1):
            # if np.array_equal(imgIR[y, x], [0,0,0]):
            disparity = disparityRL[y, x]
            # if(disparity == 0 and (x+1 < width and x-1 > -1 and y+1 < height and y-1 > -1)):
            #     disparity = max(disparityLR[y,x+1], disparityLR[y, x-1], 
            #                     disparityLR[y+1,x], disparityLR[y-1, x])
            newX = math.floor(x + (1-alpha)*disparity)
            newX = max(0, min(newX, width-1))
            disparityIR[y,newX] = (1-alpha)*disparity
            # imgIR[y, newX] = imgR_rgb[y, x]

    # Fill holes
    for y in range(height):
        for x in range(1,width-1):
            if ((np.std([disparityIR[y,x-1], disparityIR[y,x], disparityIR[y,x+1]]) > 20 or disparityIR[y,x] == -1) and np.std([disparityIR[y,x-1], disparityIR[y,x+1]]) < 3):
                disparityIR[y,x] = np.average([disparityIR[y,x-1], disparityIR[y,x+1]])

    for y in range(height):
        for x in range(width):
            if (disparityIR[y, x] >= 0):
                imgIR[y, x] = imgR_rgb[y, x - int(np.round(disparityIR[y, x]))]

    # for y in range(height):
    #     for x in range(width):
    #         # if np.array_equal(imgIL[y, x], [0,0,0]):
    #         #     disparity = -ground_truth_disparity_right[y, x]
    #         #     newX = math.floor(x - (1-alpha)*disparity)
    #         #     newX = max(0, min(newX, width-1))
    #         #     imgIR[y, x] = imgR_rgb[y, newX]
    #         disparity = disparityRL[y, x]
    #         if(disparity == 0 and (x+1 < width and x-1 > -1 and y+1 < height and y-1 > -1)):
    #             disparity = max(disparityRL[y,x+1], disparityRL[y, x-1], 
    #                             disparityRL[y+1,x], disparityRL[y-1, x])
    #         newX = math.floor(x + (1-alpha)*disparity)
    #         newX = max(0, min(newX, width-1))
    #         imgIR[y, newX] = imgR_rgb[y, x]

    # First, create a sorted list of coordinates by disparity magnitude
    # pixel_disparities = []
    # for y in range(height):
    #     for x in range(width):
    #         disparity = disparityRL[y, x]
    #         # if disparity == 0 and (x+1 < width and x-1 > -1 and y+1 < height and y-1 > -1):
    #         #     disparity = max(disparityRL[y,x+1], disparityRL[y, x-1],
    #         #                 disparityRL[y+1,x], disparityRL[y-1, x])
    #         pixel_disparities.append((abs(disparity), disparity, y, x))

    # # Sort by absolute disparity value (highest first)
    # pixel_disparities.sort(reverse=True)

    # # Process pixels in order of disparity magnitude
    # for _, disparity, y, x in pixel_disparities:
    #     newX = math.floor(x + (1-alpha)*disparity)
    #     newX = max(0, min(newX, width-1))
        
    #     # Only write if the target position is empty (zeros)
    #     if np.array_equal(imgIR[y, newX], [0,0,0]):
    #         disparityIR[y,newX] = disparity
    #         imgIR[y, newX] = imgR_rgb[y, x]


    for y in range(height):
        for x in range(width):
            if (not np.array_equal(disparityIL[y, x], [-1]) and not np.array_equal(disparityIR[y, x], [-1])):
                if (abs(disparityIL[y, x]) > abs(disparityIR[y, x])):
                    imgI[y, x] = imgIL[y, x]
                else:
                    imgI[y, x] = imgIR[y, x]
            elif (not np.array_equal(disparityIL[y, x], [-1])):
                imgI[y, x] = imgIL[y, x]
            elif (not np.array_equal(disparityIR[y, x], [-1])):
                imgI[y, x] = imgIR[y, x]

    # imgIL, imgIR, imgI = synthesize_views_cupy(imgL_rgb, imgR_rgb, ground_truth_disparity_left, ground_truth_disparity_right, alpha)

    # print("END")
    # end = time.time()
    # print(end-start)

    # #imgI = cv.fastNlMeansDenoisingColored(imgI,None,10,10,7,12)
    # Display imgL
    # plt.subplot(3, 3, 1)  # (rows, columns, index)
    # plt.imshow(imgL_rgb)
    # plt.title('Left Image')
    # plt.axis('off')

    # # Display imgR
    # plt.subplot(3, 3, 2)
    # plt.imshow(imgR_rgb)
    # plt.title('Right Image')
    # plt.axis('off')

    # plt.subplot(3, 3, 4)
    # plt.imshow(disparityIL)
    # plt.title('disparity left')
    # plt.axis('off')

    # plt.subplot(3, 3, 5)
    # plt.imshow(disparityIR)
    # plt.title('disparity right')
    # plt.axis('off')

    # # Display imgIR
    # plt.subplot(3, 3, 8)
    # plt.imshow(imgIR)
    # plt.title('Processed Image right')
    # plt.axis('off')

    # # Display imgI
    # plt.subplot(3, 3, 7)
    # plt.imshow(imgIL)
    # plt.title('Processed Image Left')
    # plt.axis('off')

    # # Display imgI
    # plt.subplot(3, 3, 9)
    # plt.imshow(imgI)
    # plt.title('Processed Image')
    # plt.axis('off')
    # plt.show()
    imgI = cv2.cvtColor(imgI, cv2.COLOR_BGR2RGB)
    global count
    filename = f"img{count}.png"
    cv2.imwrite(filename, imgI)
    count += 1




if __name__ == "__main__":
    # func('dataset/1/1/0003.png', 'dataset/1/1/0002.png')
    model_name = 'RAFT'
    func('dataset/1/2/right.png', 'dataset/1/2/left.png', 1/3, model_name)
    func('dataset/2/2/0003.png', 'dataset/2/2/0002.png', 1/3, model_name)
    func('dataset/3/2/0003.png', 'dataset/3/2/0002.png', 1/3, model_name)
    func('dataset/4/2/0003.png', 'dataset/4/2/0002.png', 1/3, model_name)
    func('dataset/5/2/0003.png', 'dataset/5/2/0002.png', 1/3, model_name)
    func('dataset/6/2/0003.png', 'dataset/6/2/0002.png', 1/3, model_name)
    func('dataset/7/right.png', 'dataset/7/left.png', 1/3, model_name)
    func('dataset/8/right.png', 'dataset/8/left.png', 1/3, model_name)
    func('dataset/9/left.png', 'dataset/9/right.png', 1/3, model_name)
    func('dataset/10/0003.png', 'dataset/10/0002.png', 1/3, model_name)
    # compare_disparities('img/2/image.npy', 'img/2/image.png')