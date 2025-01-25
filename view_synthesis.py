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
    disparityIL_helper = np.round(disparityIL)
    # # remove ghosting
    boundaryLR, diff = detect_EOBMR(disparityLR, 70, 5)
    boundaryIL = detect_EOBMV(disparityIL_helper, 5)
    combined_boundary = cv2.bitwise_and(boundaryLR, boundaryIL)

    # smooth the edges
    disparityIL = blur_boundaries(disparityIL)
    disparityIL = clean_disparity_map(disparityIL)
    # for y in range(height):
    #     for x in range(1,width-1):
    #         if ((np.std([disparityIL[y,x-1], disparityIL[y,x], disparityIL[y,x+1]]) > 20 or disparityIL[y,x] == -1) and np.std([disparityIL[y,x-1], disparityIL[y,x+1]]) < 3):
    #             disparityIL[y,x] = np.average([disparityIL[y,x-1], disparityIL[y,x+1]])

    imgIL = warp_image_cv2(imgL_rgb, disparityIL, 1, alpha)
    disparityIL[combined_boundary == 1] = -1
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
    disparityIR_helper = np.round(disparityIR)
    # remove ghosting
    boundaryRL, diff = detect_EOBMR(disparityRL, 70, 5)
    boundaryIR = detect_EOBMV(disparityIR_helper, 5)
    combined_boundary = cv2.bitwise_and(boundaryRL, boundaryIR)

    # smooth the edges
    disparityIR = blur_boundaries(disparityIR)
    disparityIR = clean_disparity_map(disparityIR)
    # for y in range(height):
    #     for x in range(1,width-1):
    #         if ((np.std([disparityIR[y,x-1], disparityIR[y,x], disparityIR[y,x+1]]) > 20 or disparityIR[y,x] == -1) and np.std([disparityIR[y,x-1], disparityIR[y,x+1]]) < 3):
    #             disparityIR[y,x] = np.average([disparityIR[y,x-1], disparityIR[y,x+1]])
    start = time.time()

    imgIR = warp_image_cv2(imgR_rgb, disparityIR, -1, (1-alpha))
    disparityIR[combined_boundary == 1] = -1
    plt.imshow(imgIR)
    plt.show()
    # for y in range(height):
    #     for x in range(width):
    #         if (disparityIR[y, x] >= 0):
    #             imgIR_copy[y, x] = imgR_rgb[y, x - int(np.round(disparityIR[y, x]))]

    # Compute the absolute difference
    # difference = cv2.absdiff(imgIR, imgIR_copy)

    # # Create a binary mask of differences
    # grayscale_diff = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    # _, binary_mask = cv2.threshold(grayscale_diff, 50, 255, cv2.THRESH_BINARY)

    # # Highlight differences in the original images
    # highlighted_diff = imgIR.copy()
    # highlighted_diff[binary_mask > 0] = [0, 0, 255]  # Highlight differences in red

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
    disparityIL = np.round(disparityIL)
    disparityIR = np.round(disparityIR)

    disparity_normalized = cv2.normalize(disparityIR, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create color disparity map
    disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    
    # Create red overlay for boundaries
    boundary_overlay = np.zeros_like(disparity_color)
    boundary_overlay[combined_boundary == 1] = [0, 0, 255]  # Red color
    
    # Combine images
    result = cv2.addWeighted(disparity_color, 1, boundary_overlay, 0.5, 0)
    
    # Display
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Disparity Map with Boundaries')
    plt.axis('off')
    plt.show()

    # remove ghosting
    # boundaryLR, diff = detect_EOBMR(disparityLR, 70, 5)
    # boundaryIL = detect_EOBMV(disparityIL, 5)
    # combined_boundary = cv2.bitwise_and(boundaryLR, boundaryIL)

    # disparityIL[combined_boundary == 1] = -1

    # boundaryRL, diff = detect_EOBMR(disparityRL, 70, 5)
    # boundaryIR = detect_EOBMV(disparityIR, 5)
    # combined_boundary = cv2.bitwise_and(boundaryRL, boundaryIR)

    # disparityIR[combined_boundary == 1] = -1
    


    check_pixels = np.zeros((imgL.shape[0], imgL.shape[1], 1), np.uint8)

    start = time.time()
    # imgI = optimize_disparity_merge(imgIL, imgIR, disparityIL, disparityIR, alpha)
    for y in range(height):
        for x in range(width):
            if (not np.array_equal(disparityIL[y, x], [-1]) and not np.array_equal(disparityIR[y, x], [-1])):
                if(not np.array_equal(disparityIL[y, x], [0]) and not np.array_equal(disparityIR[y, x], [0]) and abs(disparityIL[y, x] - disparityIR[y, x]) < 20):
                    check_pixels[y, x] = 1
                    # diff = np.mean(np.abs(imgIL[y, x] - imgIR[y, x]))
                    # if(diff > 250):
                    #     imgI[y, x] = [0,0, 255]
                    # else:
                    #     interpolated = alpha * imgIL[y, x] + (1 - alpha) * imgIR[y, x]
                    #     imgI[y, x] = interpolated.astype(np.uint8)
                    # imgI[y, x] = alpha*imgIL[y, x] + (1-alpha)*imgIR[y, x]
                    if(disparityIL[y, x]*(1-alpha) > (alpha)*disparityIR[y, x]):
                        imgI[y, x] = imgIL[y, x]
                    else:
                        imgI[y, x] = imgIR[y, x]
                elif (abs(disparityIL[y, x]) > abs(disparityIR[y, x])):
                    imgI[y, x] = imgIL[y, x]
                elif (np.array_equal(imgIR[y, x], [0, 0, 0]) and not np.array_equal(imgIL[y, x], [0, 0, 0])): # in some case bot have 0 disparities but imgIR also has black pixels
                    imgI[y, x] = imgIL[y, x]
                else:
                    imgI[y, x] = imgIR[y, x]
            elif (not np.array_equal(disparityIL[y, x], [-1])):
                imgI[y, x] = imgIL[y, x]
            elif (not np.array_equal(disparityIR[y, x], [-1])):
                imgI[y, x] = imgIR[y, x]

    print("Merging " + str(time.time() - start))
    return imgI, disparityIL, disparityIR, imgIL, imgIR


def create_edge_mask(disparity_map, mask_width=3):

    sobelx = cv2.Sobel(disparity_map, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(disparity_map, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    threshold = np.mean(magnitude) + np.std(magnitude)
    boundaries = (magnitude > threshold).astype(np.uint8) * 255

    plt.imshow(boundaries)
    plt.show()

    return boundaries, threshold

# EOBRM https://sci-hub.se/10.1109/TBC.2013.2281658
def detect_EOBMR(disparity_map, threshold=20, L1=3):
    """
    Implements EOBMR boundary detection for disparity maps
    
    Args:
        disparity_map: Input disparity map
        L1: Size of structuring element (kernel)
        chi: Threshold for depth difference
    """
    # Create structuring element B1
    B1 = np.ones((L1, L1), np.uint8)
    
    # Perform morphological dilation
    dilated = cv2.dilate(disparity_map, B1)

    # plt.subplot(1, 2, 1)  # (rows, columns, index) 
    # plt.imshow(disparity_map)
    # plt.title('normal')
    # plt.axis('off')

    # plt.subplot(1, 2, 2)  # (rows, columns, index) 
    # plt.imshow(dilated)
    # plt.title('dilated')
    # plt.axis('off')
    # plt.show()
    
    # Calculate difference between dilated and original
    diff = dilated - disparity_map
    
    # Apply threshold to get EOBMR
    EOBMR = np.where(diff > threshold, 1, 0).astype(np.uint8)

    # Additional dilation to make boundaries thicker
    EOBMR = cv2.dilate(EOBMR, np.ones((5,5), np.uint8))
    
    return EOBMR, diff

def detect_EOBMV(disparity_map, L1=3):
    """
    Detects boundaries at invalid disparity regions (EOBMV)
    """
    B1 = np.ones((L1, L1), np.uint8)
    invalid_regions = (disparity_map == -1).astype(np.uint8)
    dilated_invalid = cv2.dilate(invalid_regions, B1)
    dilated_invalid = cv2.dilate(dilated_invalid, np.ones((3,3), np.uint8))
    EOBMV = cv2.absdiff(dilated_invalid, invalid_regions)
    return EOBMV

def show_boundary(image, mask):
    # Ensure the mask is binary (if not, threshold it)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Convert mask to 3-channel (if needed)
    color_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Assign a color to the mask (e.g., red)
    color_mask[:, :, 1:] = 0  # Set green and blue channels to 0 (keep only red)

    # Blend the image and the mask
    blended = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

    plt.imshow(blended)
    plt.show()

def refine_boundaries(disparity_map, discontinuity_mask, stable_distance=2):
    refined_map = disparity_map.copy()
    
    # Iterate over all detected discontinuity pixels
    rows, cols = discontinuity_mask.shape
    for y in range(rows):
        for x in range(cols):
            if discontinuity_mask[y, x]:
                # Move to a stable region
                stable_x = max(0, x - stable_distance)
                stable_disparity = disparity_map[y, stable_x]
                
                # Update the disparity map at the boundary
                refined_map[y, x] = stable_disparity
    
    # Optional: Apply smoothing
    refined_map = cv2.bilateralFilter(refined_map.astype(np.float32), d=5, sigmaColor=50, sigmaSpace=50)

    return refined_map


# def optimize_disparity_merge(imgIL, imgIR, disparityIL, disparityIR, alpha):
#     imgI = np.zeros_like(imgIL)
    
#     # Remove extra dimensions from disparities
#     disparityIL = disparityIL.squeeze()
#     disparityIR = disparityIR.squeeze()
    
#     # Create masks
#     valid_mask = (disparityIL != -1) & (disparityIR != -1)
#     valid_disparity = (disparityIL != 0) & (disparityIR != 0)
#     disparity_diff = np.abs(disparityIL - disparityIR) < 20
    
#     condition1 = valid_mask & valid_disparity & disparity_diff
#     disparity_weight_mask = (disparityIL * alpha) < ((1 - alpha) * disparityIR)
    
    
#     # Expand masks for 3D image arrays
#     condition1 = condition1[..., np.newaxis]
#     disparity_weight_mask = disparity_weight_mask[..., np.newaxis]
    
#     # Apply conditions
#     imgI = np.where(condition1 & disparity_weight_mask, imgIL, imgI)
#     imgI = np.where(condition1 & ~disparity_weight_mask, imgIR, imgI)
    
#     # Handle remaining cases
#     disparity_compare = (np.abs(disparityIL) > np.abs(disparityIR))[..., np.newaxis]
#     black_pixels = (np.all(imgIR == [0, 0, 0], axis=-1) & ~np.all(imgIL == [0, 0, 0], axis=-1))[..., np.newaxis]
    
#     remaining_mask = valid_mask[..., np.newaxis] & ~condition1
#     imgI = np.where(remaining_mask & disparity_compare, imgIL, imgI)
#     imgI = np.where(remaining_mask & black_pixels, imgIL, imgI)
#     imgI = np.where(remaining_mask & ~disparity_compare & ~black_pixels, imgIR, imgI)
    
#     # Single valid disparity cases
#     imgI = np.where((disparityIL != -1)[..., np.newaxis] & (disparityIR == -1)[..., np.newaxis], imgIL, imgI)
#     imgI = np.where((disparityIR != -1)[..., np.newaxis] & (disparityIL == -1)[..., np.newaxis], imgIR, imgI)
    
#     return imgI

def blur_boundaries(disparity_map):
    # Create a mask for valid disparity values
    valid_mask = (disparity_map != -1).astype(np.uint8)

    # Replace -1 with neutral values (e.g., 0) for processing
    temp_disparity = np.where(disparity_map == -1, 0, disparity_map).astype(np.float32)
    if temp_disparity.ndim == 3:
        temp_disparity = temp_disparity.squeeze()

    # Apply median blur to the valid disparity region
    blurred_disparity = cv2.medianBlur(temp_disparity, 5)

    # Restore -1 values using the mask
    smoothed_disparity = np.where(valid_mask == 1, blurred_disparity[..., np.newaxis], -1)

    return smoothed_disparity

def clean_disparity_map(disparity_map, min_blob_size=100):
    # Convert to binary image
    valid_mask = (disparity_map != -1).astype(np.uint8)
    
    # Remove small blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid_mask, connectivity=8)
    
    # Filter components based on size
    for label in range(1, num_labels):  # Start from 1 to skip background
        if stats[label, cv2.CC_STAT_AREA] < min_blob_size:
            disparity_map[labels == label] = -1
            
    # Optional: Apply closing to smooth edges
    kernel = np.ones((3,3), np.uint8)
    valid_mask = (disparity_map != -1).astype(np.uint8)
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_CLOSE, kernel)
    
    # Update disparity map based on mask
    disparity_map[valid_mask == 0] = -1
    
    return disparity_map