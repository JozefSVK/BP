import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import math
import time
import PySide6.QtCore
import cupy as cp
import time
from cupyx import scatter_add
import view_synthesis
import face_morph
import utilities

count = 1





def filter_disparity_map(disparity, min_threshold=5, max_threshold=None, invalid_value=0):
    """
    Filter disparity map based on thresholds.
    
    Args:
        disparity: Input disparity map
        min_threshold: Minimum disparity value to keep
        max_threshold: Maximum disparity value to keep (optional)
        invalid_value: Value to set for filtered out pixels
    """
    filtered = disparity.copy()
    
    # Create mask for invalid values
    mask = filtered < min_threshold
    
    # Add maximum threshold if specified
    if max_threshold is not None:
        mask |= filtered > max_threshold
        
    # Set invalid values
    filtered[mask] = invalid_value
    
    return filtered

def rectify_images(img1, img2):
    # Step 1: Detect and Match Features
    orb = cv2.ORB_create()  # Or use cv2.SIFT_create() if OpenCV is built with non-free modules
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Match features using a brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Step 2: Compute the Fundamental Matrix
    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    # Select inliers
    points1 = points1[mask.ravel() == 1]
    points2 = points2[mask.ravel() == 1]

    # Step 3: Compute the Rectification Transform
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    _, H1, H2 = cv2.stereoRectifyUncalibrated(points1, points2, F, imgSize=(w1, h1))

    # Step 4: Warp the Images
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))

    return img1_rectified, img2_rectified

def remove_pixels(img, disparity, threshold=0):
    # Create mask where disparity exists (non-zero)
    mask = disparity <= threshold
    
    # Convert to 3-channel mask for color image
    mask = np.stack([mask] * 3, axis=2)
    
    # Apply mask to image
    masked_image = np.where(mask, img, 0)
    
    return masked_image

def func(left_image_path, right_image_path, donwscale=0.5, model='IGEV', top_down_imgs = False):
    wholeTime = time.time()
    print('Starting')
    imgL = cv2.imread(left_image_path)
    imgR = cv2.imread(right_image_path)

    # downscale
    imgL = cv2.resize(imgL, None, fx=donwscale, fy=donwscale)
    imgR = cv2.resize(imgR, None, fx=donwscale, fy=donwscale)
    height, width = imgR.shape[:2]
    if(top_down_imgs):
        imgL = cv2.rotate(imgL, cv2.ROTATE_90_CLOCKWISE)
        imgR = cv2.rotate(imgR, cv2.ROTATE_90_CLOCKWISE)
    # print(imgL.shape[:2])

    # rect_imgL, rect_imgR = rectify_images(imgL, imgR)

    # plt.subplot(2, 2, 1)  # (rows, columns, index)
    # plt.imshow(imgL)
    # plt.title('Left Image')
    # plt.axis('off')

    # plt.subplot(2, 2, 2)  # (rows, columns, index)
    # plt.imshow(imgR)
    # plt.title('Right Image')
    # plt.axis('off')

    # plt.subplot(2, 2, 3)  # (rows, columns, index)
    # plt.imshow(rect_imgL)
    # plt.title('Right Image')
    # plt.axis('off')

    # plt.subplot(2, 2, 4)  # (rows, columns, index)
    # plt.imshow(rect_imgR)
    # plt.title('Right Image')
    # plt.axis('off')
    # plt.show()
    # return
    # call wrapper
    start = time.time()
    disparityLR, disparityRL = utilities.calculate_disparity(imgL, imgR, model)
    print("disparity calculator " + str(time.time() - start))

    plt.imshow(disparityLR)
    plt.show()
    # save_disparity(disparityLR, 'background.png')
    filter_value = 130
    # disparityLR = filter_disparity_map(disparityLR, filter_value, None, 0)
    # disparityRL = filter_disparity_map(disparityRL, filter_value, None, 0)
    # save_disparity(disparityLR, 'dataset_comparison/raft_realtime.png')
    plt.imshow(disparityRL)
    plt.show()
    # return
    # start = time.time()
    # boundary_mask, depth_diff = view_synthesis.detect_EOBMR(disparityLR, 70, 5)
    # print("boundary" + str(time.time() - start))
    # plt.imshow(boundary_mask)
    # plt.show()
    # plt.imshow(depth_diff)
    # plt.show()


    # # Option 2: Automatic thresholding using Otsu's method
    # disparityLR_8bit = cv2.normalize(disparityLR, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # _, foreground_mask_otsu = cv2.threshold(disparityLR_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Clean up the mask using morphological operations
    # kernel = np.ones((5, 5), np.uint8)
    # foreground_mask_cleaned = cv2.morphologyEx(foreground_mask_otsu, cv2.MORPH_OPEN, kernel)
    # foreground_mask_cleaned = cv2.morphologyEx(foreground_mask_cleaned, cv2.MORPH_CLOSE, kernel)

    # # Extract foreground from original image (if needed)
    # foreground_only = cv2.bitwise_and(disparityLR, disparityLR, mask=foreground_mask_cleaned)

    # plt.imshow(foreground_only)
    # plt.show()

    


    imgL_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    imgR_rgb = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

    # new_img = remove_pixels(imgL_rgb, disparityLR)
    # plt.imshow(new_img)
    # plt.show()

    imgIR = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.uint8)
    imgIL = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.uint8)
    imgI = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.uint8)


    alpha = 0.5
    # start = time.time()
    # print("Starting")

    disparityIL = np.ones((imgL.shape[0], imgL.shape[1], 1), np.float32) * -1
    disparityIR = np.ones((imgL.shape[0], imgL.shape[1], 1), np.float32) * -1
    """
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
    """
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

    """
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
    """
    # imgIL, imgIR, imgI = synthesize_views_cupy(imgL_rgb, imgR_rgb, ground_truth_disparity_left, ground_truth_disparity_right, alpha)

    # print("END")
    # end = time.time()
    # print(end-start)

    start = time.time()
    # imgI, disparityIL, disparityIR, imgIL, imgIR  = GPU_intermediate_view(imgL, imgR, disparityLR, disparityRL, 0.5)

    
    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files
    # appendFilename = "topDown" if top_down_imgs else "leftRight"
    # filename = f"img_{model}_{appendFilename}.mp4"
    # video_writer = cv2.VideoWriter(filename, fourcc, 20, (width, height))
    # inter_values = np.linspace(0, 1, 101)
    # for value in inter_values:
    #     value = round(value, 2)
    #     print("value " + str(value))
    #     imgI, disparityIL, disparityIR, imgIL, imgIR  = view_synthesis.create_intermediate_view(imgL, imgR, disparityLR, disparityRL, value)
    #     imgI = cv2.cvtColor(imgI, cv2.COLOR_BGR2RGB)
    #     if(top_down_imgs):
    #         imgI = cv2.rotate(imgI, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     # filename = f"img_{model}_{value}.png"
    #     # cv2.imwrite(filename, imgI)
    #     video_writer.write(imgI)
    # video_writer.release()


    imgI, disparityIL, disparityIR, imgIL, imgIR  = view_synthesis.create_intermediate_view(imgL, imgR, disparityLR, disparityRL, alpha)
    # print("Calculation of middle " + str(time.time() - start))
    print("Whole time " + str(time.time() - start))
    # #imgI = cv.fastNlMeansDenoisingColored(imgI,None,10,10,7,12)
    # Display imgL

    # Detect landmarks
    # landmarks_left = face_morph.detect_landmarks(imgL_rgb)
    # landmarks_right = face_morph.detect_landmarks(imgR_rgb)
    # if landmarks_left is None or landmarks_right is None:
    #     print("Failed to detect landmarks")
    # else:
    #     # Compute intermediate landmarks
    #     intermediate_landmarks = face_morph.compute_intermediate_landmarks(landmarks_left, landmarks_right)
        
    #     # Perform Delaunay triangulation
    #     delaunay = face_morph.compute_delaunay_triangulation(intermediate_landmarks)
        
    #     # Warp both images to the intermediate view
    #     warped_left, warped_right = face_morph.warp_images_to_intermediate(
    #         imgL_rgb, imgR_rgb, landmarks_left, landmarks_right, intermediate_landmarks, delaunay
    #     )
        
    #     # Blend the warped images
    #     synthesized_image = face_morph.alpha_blend(warped_left, warped_right)

    #     imgI = face_morph.merge_image(synthesized_image, imgI)


    # rotate iamges if top down view
    if(top_down_imgs):
        imgI = cv2.rotate(imgI, cv2.ROTATE_90_COUNTERCLOCKWISE)
        imgL_rgb = cv2.rotate(imgL_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        imgR_rgb = cv2.rotate(imgR_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        disparityIL = cv2.rotate(disparityIL, cv2.ROTATE_90_COUNTERCLOCKWISE)
        disparityIR = cv2.rotate(disparityIR, cv2.ROTATE_90_COUNTERCLOCKWISE)
        imgIR = cv2.rotate(imgIR, cv2.ROTATE_90_COUNTERCLOCKWISE)
        imgIL = cv2.rotate(imgIL, cv2.ROTATE_90_COUNTERCLOCKWISE)

    
    # save_disparity(disparityLR, "LR_disparity.png")
    # save_disparity(disparityRL, "RL_disparity.png")
    # save_disparity(disparityIL, "IL_disparity.png")
    # save_disparity(disparityIR, "IR_disparity.png")

    # dispaly images
    plt.subplot(3, 3, 1)  # (rows, columns, index) 
    plt.imshow(imgL_rgb)
    plt.title('Left Image')
    plt.axis('off')

    # Display imgR
    plt.subplot(3, 3, 2)
    plt.imshow(imgR_rgb)
    plt.title('Right Image')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(disparityIL)
    plt.title('disparity left')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(disparityIR)
    plt.title('disparity right')
    plt.axis('off')

    # Display imgIR
    plt.subplot(3, 3, 8)
    plt.imshow(imgIR)
    plt.title('Processed Image right')
    plt.axis('off')

    # Display imgI
    plt.subplot(3, 3, 7)
    plt.imshow(imgIL)
    plt.title('Processed Image Left')
    plt.axis('off')

    # Display imgI
    plt.subplot(3, 3, 9)
    plt.imshow(imgI)
    plt.title('Processed Image')
    plt.axis('off')
    plt.show()
    imgI = cv2.cvtColor(imgI, cv2.COLOR_BGR2RGB)
    # imgIL = cv2.cvtColor(imgIL, cv2.COLOR_BGR2RGB)
    # imgIR = cv2.cvtColor(imgIR, cv2.COLOR_BGR2RGB)
    # global count
    # filename = f"img{count}.png"
    # cv2.imwrite(filename, imgI)
    # count += 1
    # cv2.imwrite("R5.png", imgI)
    # cv2.imwrite("Middle_L.png", imgIL)
    # cv2.imwrite("Middle_R.png", imgIR)


def save_disparity(disparity_map, text):
    disparity_normalized = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min())

    # Scale to 0-255 and convert to uint8
    disparity_uint8 = (disparity_normalized * 255).astype(np.uint8)
    cv2.imwrite(text, disparity_uint8)

if __name__ == "__main__":
    # func('dataset/1/1/0003.png', 'dataset/1/1/0002.png')
    model_name = 'IGEV'
    top_down = True
    # func('dataset/1/2/right.png', 'dataset/1/2/left.png', 1/3, model_name)
    # func('dataset/real/topL.png', 'dataset/real/topR.png', 1, model_name)
    # func('dataset/real/botL.png', 'dataset/real/topL.png', 1, model_name, top_down)
    
    # func('view1.png', 'view5.png', 1, model_name)
    # func('im0.png', 'im1.png', 1/2, model_name)
    # func('dataset/1/2/0004.png', 'dataset/1/2/0005.png', 1/3, model_name, top_down)
    # func('dataset/3/2/0003.png', 'dataset/3/2/0002.png', 1/3, model_name)
    # func('dataset/4/2/0003.png', 'dataset/4/2/0002.png', 1/3, model_name)
    # func('dataset/5/2/0003.png', 'dataset/5/2/0002.png', 1/3, model_name)
    # func('dataset/6/2/0003.png', 'dataset/6/2/0002.png', 1/3, model_name)
    # func('dataset/7/right.png', 'dataset/7/left.png', 1/3, model_name)
    # func('dataset/8/right.png', 'dataset/8/left.png', 1/3, model_name)
    # func('dataset/9/left.png', 'dataset/9/right.png', 1/3, model_name)
    # func('dataset/2/0003.png', 'dataset/2/0002.png', 1/3, model_name)
    # func('dataset/11/0003.png', 'dataset/11/0002.png', 1/3, model_name)
    # compare_disparities('img/2/image.npy', 'img/2/image.png')

    # custom dataset
    # func('dataset/custom_dataset/bg/BL.jpeg', 'dataset/custom_dataset/bg/TL.jpeg', 1/3, model_name, top_down)
    # func('dataset/custom_dataset/bg/BR.jpeg', 'dataset/custom_dataset/bg/TR.jpeg', 1/3, model_name, top_down)
    # func('dataset/custom_res/R25.png', 'dataset/custom_res/L25.png', 1, model_name)
    # func('dataset/dataset/res/L5.png', 'dataset/dataset/res/R5.png', 1, model_name)
    # func('dataset/custom_res/R75.png', 'dataset/custom_res/L75.png', 1, model_name)
    # func('dataset/custom/botL.jpeg', 'dataset/custom/botR.jpeg', 1/3, model_name)
    func('dataset/custom_dataset/bg/BL.jpeg', 'dataset/custom_dataset/bg/BR.jpeg', 1/3, model_name)
    # func('dataset/custom_dataset/bg/TL.jpeg', 'dataset/custom_dataset/bg/TR.jpeg', 1/3, model_name)

