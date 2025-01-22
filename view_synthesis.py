import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
import cupy as cp
from cupyx import scatter_add
import math


def GPU_intermediate_view(imgL, imgR, disparityLR, disparityRL, alpha = 0.5):
    # Convert images to RGB (CPU operation)
    imgL_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    imgR_rgb = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

    # Transfer data to GPU
    imgL_rgb_gpu = cp.array(imgL_rgb)
    imgR_rgb_gpu = cp.array(imgR_rgb)
    disparityLR_gpu = cp.array(disparityLR)
    disparityRL_gpu = cp.array(disparityRL)

    height, width = imgL.shape[:2]

    imgIL = cp.zeros((height, width, 3), dtype=cp.uint8)
    imgIR = cp.zeros((height, width, 3), dtype=cp.uint8)
    imgI = cp.zeros((height, width, 3), dtype=cp.uint8)

    disparityIL = cp.full((height, width), -1, dtype=cp.float32)
    disparityIR = cp.full((height, width), -1, dtype=cp.float32)

    # Calculate disparityIL
    x_coords = cp.arange(width)
    y_coords = cp.arange(height)
    xv, yv = cp.meshgrid(x_coords, y_coords)

    new_x_IL = cp.clip(cp.floor(xv - alpha * disparityLR_gpu).astype(cp.int32), 0, width - 1)
    disparityIL[yv, new_x_IL] = alpha * disparityLR_gpu[yv, xv]

    plt.imshow(cp.asnumpy(disparityIL))
    plt.show()

    # Fill holes in disparityIL using a median filter
    for i in range(3):  # Perform iterative smoothing
        mask_invalid = (disparityIL == -1)
        disparityIL = cp.where(
            mask_invalid,
            (cp.roll(disparityIL, 1, axis=1) + cp.roll(disparityIL, -1, axis=1)) / 2,
            disparityIL
        )

    # Populate imgIL
    valid_disp_IL = disparityIL >= 0
    x_offset_IL = cp.round(disparityIL).astype(cp.int32)
    valid_x_IL = cp.clip(xv + x_offset_IL, 0, width - 1)
    imgIL[yv, xv] = cp.where(valid_disp_IL[:, :, None], imgL_rgb_gpu[yv, valid_x_IL], 0)

    # Calculate disparityIR
    new_x_IR = cp.clip(cp.floor(xv + (1 - alpha) * disparityRL_gpu).astype(cp.int32), 0, width - 1)
    disparityIR[yv, new_x_IR] = cp.where(disparityRL_gpu >= 0, (1 - alpha) * disparityRL_gpu[yv, xv], -1)

    # Fill holes in disparityIR using a median filter
    for i in range(3):  # Perform iterative smoothing
        mask_invalid = (disparityIR == -1)
        disparityIR = cp.where(
            mask_invalid,
            (cp.roll(disparityIR, 1, axis=1) + cp.roll(disparityIR, -1, axis=1)) / 2,
            disparityIR
        )

    # Populate imgIR
    valid_disp_IR = disparityIR >= 0
    x_offset_IR = cp.round(disparityIR).astype(cp.int32)
    valid_x_IR = cp.clip(xv - x_offset_IR, 0, width - 1)
    imgIR[yv, xv] = cp.where(valid_disp_IR[:, :, None], imgR_rgb_gpu[yv, valid_x_IR], 0)

    # Combine imgIL and imgIR into imgI
    valid_IL = disparityIL >= 0
    valid_IR = disparityIR >= 0
    use_IL = cp.logical_and(valid_IL, ~valid_IR) | (cp.abs(disparityIL) <= cp.abs(disparityIR))
    imgI = cp.where(use_IL[:, :, None], imgIL, imgIR)

    # Transfer result back to CPU
    imgI_cpu = cp.asnumpy(imgI)
    disparityIL_cpu = cp.asnumpy(disparityIL)
    disparityIR_cpu = cp.asnumpy(disparityIR)
    imgIL_cpu = cp.asnumpy(imgIL)
    imgIR_cpu = cp.asnumpy(imgIR)
    return imgI_cpu, disparityIL_cpu, disparityIR_cpu, imgIL_cpu, imgIR_cpu


def fill_disparity_holes_gpu(disparity_map, std_threshold1=20, std_threshold2=3):
    # Transfer to GPU if not already there
    if isinstance(disparity_map, np.ndarray):
        disparity_map = cp.asarray(disparity_map)
    
    # Handle 3D input
    is_3d = disparity_map.ndim == 3
    if is_3d:
        disparity_map = disparity_map.squeeze()
    
    # Create rolling windows of size 3
    window = cp.lib.stride_tricks.sliding_window_view(disparity_map, 3, axis=1)
    
    # Calculate standard deviation for each window
    std_window = cp.std(window, axis=2)
    
    # Calculate standard deviation of neighbors (excluding center)
    neighbors = cp.stack([window[:,:,0], window[:,:,2]], axis=2)
    std_neighbors = cp.std(neighbors, axis=2)
    
    # Create mask for points that need filling
    center_invalid = disparity_map[:, 1:-1] == -1
    high_std = std_window > std_threshold1
    low_neighbor_std = std_neighbors < std_threshold2
    fill_mask = (high_std | center_invalid) & low_neighbor_std
    
    # Calculate average of neighbors where needed
    avg_neighbors = cp.mean(neighbors, axis=2)
    
    # Fill holes
    disparity_map[:, 1:-1][fill_mask] = avg_neighbors[fill_mask]
    
    # Restore 3D shape if needed
    if is_3d:
        disparity_map = disparity_map[..., None]
    
    # Return as numpy array
    return cp.asnumpy(disparity_map)


def warp_disparity_gpu(disparityLR, alpha, height, width):
    disparityLR = cp.asarray(disparityLR)
    disparityIL = cp.zeros_like(disparityLR)
    
    # Create grid for x and y indices
    y_indices, x_indices = cp.meshgrid(cp.arange(height), cp.arange(width), indexing='ij')
    
    # Compute new x positions
    disparity = disparityLR[y_indices, x_indices]
    newX = cp.floor(x_indices - alpha * disparity)
    newX = cp.clip(newX, 0, width - 1).astype(cp.int32)
    
    # Calculate the new value
    new_values = alpha * disparity

    # Iterate to ensure updates only occur when the new value is greater
    for y in range(height):
        for x in range(width):
            nx = newX[y, x]
            if new_values[y, x] > disparityIL[y, nx]:
                disparityIL[y, nx] = new_values[y, x]
    
    return cp.asnumpy(disparityIL)


def warp_image_cv2(img_rgb, disparityI, multiplier, alpha):
    height, width = img_rgb.shape[:2]
    
    # Create coordinate maps
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Ensure disparity is 2D by squeezing
    if disparityI.ndim == 3:
        disparityI = disparityI.squeeze()
    
    # Calculate source x coordinates
    x_map = x_coords + (multiplier*alpha*np.round(disparityI))
    
    # Create valid mask for non-negative disparities
    valid_mask = (disparityI >= 0)
    
    # Set invalid positions to 0 (or any other value that will be masked out)
    x_map[~valid_mask] = 0
    
    # Ensure coordinates are within bounds
    x_map = np.clip(x_map, 0, width - 1)
    
    # Create remapping matrices
    map_x = x_map.astype(np.float32)
    map_y = y_coords.astype(np.float32)
    
    # Apply remapping
    imgI = cv2.remap(img_rgb, map_x, map_y, cv2.INTER_LINEAR)
    
    # Mask out invalid pixels
    imgI[~valid_mask] = 0
    
    return imgI

def combine_images_gpu(imgIL, imgIR, disparityIL, disparityIR):
    # Ensure disparities are 2D
    if disparityIL.ndim == 3:
        disparityIL = disparityIL.squeeze()
    if disparityIR.ndim == 3:
        disparityIR = disparityIR.squeeze()

    # Create masks for valid disparity values
    valid_L = (disparityIL != -1)
    valid_R = (disparityIR != -1)
    
    # Create mask for black pixels in imgIR
    black_R = np.all(imgIR == 0, axis=2)
    non_black_L = ~np.all(imgIL == 0, axis=2)
    
    # Create mask for larger disparity
    larger_L = np.abs(disparityIL) < np.abs(disparityIR)
    
    # Initialize output with zeros
    imgI = np.zeros_like(imgIL)
    
    # Create the final masks
    use_L = np.zeros_like(valid_L, dtype=bool)
    
    # Both disparities valid
    both_valid = valid_L & valid_R
    use_L[both_valid] = (
        larger_L[both_valid] | 
        (black_R[both_valid] & non_black_L[both_valid])
    )
    
    # Single disparity valid
    use_L[~both_valid] = valid_L[~both_valid]
    
    # Use OpenCV to copy pixels based on masks
    imgI[use_L] = imgIL[use_L]
    imgI[~use_L & valid_R] = imgIR[~use_L & valid_R]
    
    return imgI

def generate_intermediate_disparity(disparityLR, alpha, multiplier):
    height, width = disparityLR.shape

    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Calculate new x coordinates for all pixels at once
    newX = np.floor(x_coords + multiplier * alpha * disparityLR).astype(np.int32)
    
    # Clip coordinates to valid range
    newX = np.clip(newX, 0, width-1)
    
    # Initialize output array
    disparityIL = np.ones((height, width), dtype=disparityLR.dtype) * -1
    
    # Create linear indices for fast assignment
    indices = y_coords * width + newX
    
    # Handle collisions by using maximum disparity
    order = np.argsort(disparityLR.ravel())  # Sort by disparity values
    flat_indices = indices.ravel()[order]
    flat_disparities = disparityLR.ravel()[order]
    
    # Fast assignment using linear indices
    disparityIL.ravel()[flat_indices] = flat_disparities
    
    return disparityIL

def create_intermediate_view(imgL, imgR, disparityLR, disparityRL, alpha = 0.5):
    imgL_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    imgR_rgb = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)
    if(alpha == 0):
        return imgL_rgb, disparityLR, disparityRL, imgL, imgR
    elif (alpha == 1):
        return imgR_rgb, disparityLR, disparityRL, imgL, imgR

    imgIR = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.uint8)
    imgIR_copy = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.uint8)
    imgIL = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.uint8)
    imgI = np.zeros((imgL.shape[0], imgL.shape[1], 3), np.uint8)

    height, width = imgR.shape[:2]

    disparityIL = np.ones((imgL.shape[0], imgL.shape[1], 1), np.float32) * -1
    disparityIR = np.ones((imgL.shape[0], imgL.shape[1], 1), np.float32) * -1

    # disparityIL = warp_disparity_gpu(disparityLR, alpha, height, width)
    start = time.time()
    # disparityIL = generate_intermediate_disparity(disparityLR, alpha, -1)
    for y in range(height):
        for x in range(width):
            disparity = disparityLR[y, x]
            newX = math.floor(x - alpha*disparity)
            newX = max(0, min(newX, width-1))
            disparityIL[y,newX] = disparity
    # print("Intermediate disparity " + str(time.time() - start))

    # Fill holes
    disparityIL = fill_disparity_holes_gpu(disparityIL, 20, 3)
    # for y in range(height):
    #     for x in range(1,width-1):
    #         if ((np.std([disparityIL[y,x-1], disparityIL[y,x], disparityIL[y,x+1]]) > 20 or disparityIL[y,x] == -1) and np.std([disparityIL[y,x-1], disparityIL[y,x+1]]) < 3):
    #             disparityIL[y,x] = np.average([disparityIL[y,x-1], disparityIL[y,x+1]])

    imgIL = warp_image_cv2(imgL_rgb, disparityIL, 1, alpha)
    # for y in range(height):
    #     for x in range(width):
    #         if (disparityIL[y, x] >= 0):
    #             imgIL[y, x] = imgL_rgb[y, x + int(np.round(disparityIL[y, x]))]

    for y in range(height):
        for x in range(width-1, -1, -1):
            disparity = disparityRL[y, x]
            newX = math.floor(x + (1-alpha)*disparity)
            newX = max(0, min(newX, width-1))
            disparityIR[y,newX] = disparity

    # Fill holes
    disparityIR = fill_disparity_holes_gpu(disparityIR, 20, 3)
    # for y in range(height):
    #     for x in range(1,width-1):
    #         if ((np.std([disparityIR[y,x-1], disparityIR[y,x], disparityIR[y,x+1]]) > 20 or disparityIR[y,x] == -1) and np.std([disparityIR[y,x-1], disparityIR[y,x+1]]) < 3):
    #             disparityIR[y,x] = np.average([disparityIR[y,x-1], disparityIR[y,x+1]])
    start = time.time()
    imgIR = warp_image_cv2(imgR_rgb, disparityIR, -1, (1-alpha))
    # for y in range(height):
    #     for x in range(width):
    #         if (disparityIR[y, x] >= 0):
    #             imgIR_copy[y, x] = imgR_rgb[y, x - int(np.round(disparityIR[y, x]))]

    # Compute the absolute difference
    difference = cv2.absdiff(imgIR, imgIR_copy)

    # Create a binary mask of differences
    grayscale_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(grayscale_diff, 50, 255, cv2.THRESH_BINARY)

    # Highlight differences in the original images
    highlighted_diff = imgIR.copy()
    highlighted_diff[binary_mask > 0] = [0, 0, 255]  # Highlight differences in red

    # plt.subplot(2, 2, 3)
    # plt.title("Difference (Grayscale)")
    # plt.imshow(grayscale_diff, cmap='gray')
    # plt.axis("off")

    # plt.subplot(2, 2, 4)
    # plt.title("Highlighted Differences")
    # plt.imshow(cv2.cvtColor(highlighted_diff, cv2.COLOR_BGR2RGB))
    # plt.axis("off")

    # plt.tight_layout()
    # plt.show()

    # round disparities
    disparityIL = np.round(disparityIL).astype(int)
    disparityIR = np.round(disparityIR).astype(int)

    # imgI = combine_images_gpu(imgIL, imgIR, disparityIL, disparityIR)
    for y in range(height):
        for x in range(width):
            if (not np.array_equal(disparityIL[y, x], [-1]) and not np.array_equal(disparityIR[y, x], [-1])):
                if(abs(disparityIL[y, x] - disparityIR[y, x]) < 15):
                    if(disparityIL[y, x]*alpha > (1-alpha)*disparityIR[y, x]):
                        imgI[y, x] = imgIL[y, x]
                    else:
                        imgI[y, x] = imgIR[y, x]
                if (abs(disparityIL[y, x]) > abs(disparityIR[y, x])):
                    imgI[y, x] = imgIL[y, x]
                elif (np.array_equal(imgIR[y, x], [0, 0, 0]) and not np.array_equal(imgIL[y, x], [0, 0, 0])): # in some case bot have 0 disparities but imgIR also has black pixels
                    imgI[y, x] = imgIL[y, x]
                else:
                    imgI[y, x] = imgIR[y, x]
            elif (not np.array_equal(disparityIL[y, x], [-1])):
                imgI[y, x] = imgIL[y, x]
            elif (not np.array_equal(disparityIR[y, x], [-1])):
                imgI[y, x] = imgIR[y, x]

    return imgI, disparityIL, disparityIR, imgIL, imgIR