import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
import cupy as cp
from cupyx import scatter_add
import math
import MediapipeTriangulation
import MediaPipeUtilities


def get_foreground_disparity(disparity):
    disparity_8bit = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, foreground_mask_otsu = cv2.threshold(disparity_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    foreground_mask_cleaned = cv2.morphologyEx(foreground_mask_otsu, cv2.MORPH_OPEN, kernel)
    foreground_mask_cleaned = cv2.morphologyEx(foreground_mask_cleaned, cv2.MORPH_CLOSE, kernel)

    # Extract foreground from original image (if needed)
    foreground_only = cv2.bitwise_and(disparity, disparity, mask=foreground_mask_cleaned)

    return foreground_only


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

    # plt.imshow(cp.asnumpy(disparityIL))
    # plt.show()

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

def simple_crack_filling(disparityI):
    height, width = disparityI.shape[:2]
    for y in range(height):
        for x in range(1,width-1):
            if((disparityI[y,x-1] == -1 or disparityI[y,x+1] == -1) and not 
                (disparityI[y,x+1] == -1 and disparityI[y,x-1] == -1)): continue

            if ((np.std([disparityI[y,x-1], disparityI[y,x], disparityI[y,x+1]]) > 20 or 
                disparityI[y,x] == -1) and 
                np.std([disparityI[y,x-1], disparityI[y,x+1]]) < 3):
                disparityI[y,x] = np.average([disparityI[y,x-1], disparityI[y,x+1]])

    return disparityI

def fill_disparity_holes_gpu(disparity_map, std_threshold1=20, std_threshold2=3):
    # Transfer to GPU if not already there
    if isinstance(disparity_map, np.ndarray):
        disparity_map = cp.asarray(disparity_map)
    
    # Handle 3D input
    is_3d = disparity_map.ndim == 3
    if is_3d:
        disparity_map = disparity_map.squeeze()
    
    # Get left and right neighbors
    left = disparity_map[:, :-2]
    center = disparity_map[:, 1:-1]
    right = disparity_map[:, 2:]

    # Create mask where both neighbors are either valid or both invalid
    both_neighbors_valid = (left != -1) & (right != -1)
    both_neighbors_invalid = (left == -1) & (right == -1)
    use_mask = both_neighbors_valid | both_neighbors_invalid

    # Compute std of full window and of neighbors
    window = cp.stack([left, center, right], axis=2)
    std_window = cp.std(window, axis=2)
    std_neighbors = cp.std(cp.stack([left, right], axis=2), axis=2)
    avg_neighbors = cp.mean(cp.stack([left, right], axis=2), axis=2)

    # Conditions
    center_invalid = center == -1
    high_std = std_window > std_threshold1
    low_neighbor_std = std_neighbors < std_threshold2
    fill_mask = use_mask & (high_std | center_invalid) & low_neighbor_std

    # Apply correction
    center[fill_mask] = avg_neighbors[fill_mask]

    # Restore 3D shape if needed
    if is_3d:
        disparity_map = disparity_map[..., None]

    return cp.asnumpy(disparity_map)

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

def backward_warping(img_rgb, disparityI, multiplier, alpha):
    height, width = img_rgb.shape[:2]
    imgI = np.zeros((height, width, 3), np.uint8)
    for y in range(height):
        for x in range(width):
            if (disparityI[y, x] >= 0):
                xPos = (x + multiplier*int(np.round(alpha*disparityI[y, x])))
                if  -1 < xPos and xPos < width:
                    imgI[y, x] = img_rgb[y, xPos]

    return imgI


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

def forward_warping(disparityMap, alpha, width, height, is_left_disparity):
    disparityI = np.ones((height, width, 1), np.float32) * -1
    for y in range(height):
        for x in range(width):
            if disparityMap[y, x] > -1:
                disparity = disparityMap[y, x]
                if is_left_disparity: # left disparity map
                    newX = math.floor(x - alpha*disparity)
                else: # right disparity map
                    newX = math.floor(x + (1-alpha)*disparity)
                # skip invalid value
                if newX < 0 or newX >= width: continue
                
                if(disparityI[y,newX] == -1 or disparityI[y,newX] < disparity):
                    disparityI[y,newX] = disparity

    return disparityI


def create_intermediate_view(imgL, imgR, disparityLR, disparityRL, alpha = 0.5, top_down_imgs = False):
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

    disparityIL = np.ones((height, width, 1), np.float32) * -1
    disparityIR = np.ones((height, width, 1), np.float32) * -1

    # --------------------
    # Detect face in image
    # if there is one we will use custom generator for head and replace the version that DIBR generate
    # --------------------
    face_imgL = imgL
    face_imgR = imgR
    if(top_down_imgs):
        face_imgL = cv2.rotate(face_imgL, cv2.ROTATE_90_COUNTERCLOCKWISE)
        face_imgR = cv2.rotate(face_imgR, cv2.ROTATE_90_COUNTERCLOCKWISE)

    faceL = MediapipeTriangulation.get_face_landmarks(face_imgL)
    faceR = MediapipeTriangulation.get_face_landmarks(face_imgR)

    # faceL = None
    # faceR = None
    
    if faceL is not None and faceR is not None:
        # plt.imshow(cv2.cvtColor(face_imgL, cv2.COLOR_BGR2RGB))
        # plt.scatter(faceL[:, 0], faceL[:, 1], s=1, c='lime')
        # plt.title("MediaPipe – tvárové body")
        # plt.axis('off')
        # plt.savefig("obr_mediapipe_body.png", dpi=300, bbox_inches='tight')
        # plt.show()

        def get_head(face_img, face_points):
            person_mask = MediaPipeUtilities.segment_head(face_img)

            contour_mask = (person_mask > 0.5).astype(np.uint8) * 255
            person_img = cv2.bitwise_and(face_img, face_img, mask=contour_mask)

            chin_y = face_points[152][1]

            # 2) build a mask that is “1” only above the chin
            h, w = face_img.shape[:2]
            head_mask = np.zeros((h, w), dtype=np.uint8)
            head_mask[:chin_y+5, :] = 1  # +10px for safety


            # 3) optional: smooth the top edge so it isn’t jaggy
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
            head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, kernel)

            res = cv2.bitwise_and(person_img, person_img, mask=head_mask)

            head_mask = cv2.bitwise_and(contour_mask, head_mask)
            
            return res, head_mask, contour_mask
        
        # get only the head region from image       
        headL, maskL, contourL = get_head(face_imgL, faceL)
        headR, maskR, contourR = get_head(face_imgR, faceR)

        def head_contours(mask, n_points=200):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                return None
            cnt = max(cnts, key=lambda c: cv2.contourArea(c))[:,0,:]
            idx = np.round(np.linspace(0, len(cnt)-1, n_points)).astype(int)

            return cnt[idx]

        # map the edges of the head
        cL = head_contours(maskL)
        cR = head_contours(maskR)

        # combine the face and head keypoints for each image
        ptsL = np.vstack([faceL, cL])
        ptsR = np.vstack([faceR, cR])
        pts_mid = ((1-alpha)*ptsL + alpha*ptsR).astype(np.int32)

        # plt.imshow(cv2.cvtColor(face_imgL, cv2.COLOR_BGR2RGB))
        # plt.scatter(ptsL[:, 0], ptsL[:, 1], s=1, c='orange')
        # plt.title("Body tváre + kontúra hlavy")
        # plt.axis('off')
        # plt.savefig("obr_mediapipe_kontura.png", dpi=300, bbox_inches='tight')
        # plt.show()

        mask_mid = np.zeros(headL.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_mid, [pts_mid], 255)

        warp1, warp2 = MediapipeTriangulation.warp_images(headL, headR, ptsL, ptsR, pts_mid)

        if not top_down_imgs:
            mid_coord = int(np.median(pts_mid[:,0]))  # legacy left/right
        else:
            mid_coord = int(np.median(pts_mid[:,1]))  # new up/down
        Wmask = MediapipeTriangulation.compute_weight_mask(headL.shape, mid_coord, 40, "y" if top_down_imgs else "x")
        blended = cv2.convertScaleAbs(warp1*(1-Wmask) + warp2*Wmask) 
        blended_mask = np.any(blended > 0, axis=2)  
        
        # get the edges of head that aren't in the mask and remove them from disparity
        # Create structuring element B1
        B1 = np.ones((10, 10), np.uint8)
        # Perform morphological dilation
        dilatedL = cv2.dilate(maskL, B1)
        outlineL = dilatedL - maskL
        diffL = cv2.bitwise_and(outlineL, cv2.bitwise_not(contourL))
        diffL = diffL + maskL
        
        dilatedR = cv2.dilate(maskR, B1)
        outlineR = dilatedR - maskR
        diffR = cv2.bitwise_and(outlineR, cv2.bitwise_not(contourR))
        diffR = diffR + maskR

        # disparityLR[diffL > 0] = -1
        # disparityRL[diffR > 0] = -1

        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)  

        # # Vytvor biele pozadie rovnakej veľkosti
        # white_bg = np.ones_like(blended) * 255  # RGB biela

        # # Vytvor binárnu masku podľa toho, kde má blended informáciu (napr. nie je úplne čierny pixel)
        # valid_mask = np.any(blended > 0, axis=2)

        # # Vlož blended do bieleho pozadia
        # output = white_bg.copy()
        # output[valid_mask] = blended[valid_mask]

        # ground_truth = cv2.imread("dataset/1/2/0001.png")
        # ground_truth = cv2.resize(ground_truth, None, fx=1/3, fy=1/3)
        # blended_valid = blended[valid_mask]  
        # ground_truth_valid = ground_truth[valid_mask]  # shape (N, 3)

        # # príklad 1: výpočet priemernej absolútnej odchýlky (L1 rozdiel)
        # mae = np.mean(np.abs(blended_valid.astype(np.float32) - ground_truth_valid.astype(np.float32)))
        # print(f"Priemerná absolútna odchýlka: {mae:.2f}")

        # # príklad 2: MSE alebo PSNR
        # mse = np.mean((blended_valid.astype(np.float32) - ground_truth_valid.astype(np.float32))**2)
        # print(f"MSE: {mse:.2f}")

        # # (voliteľne) PSNR
        # psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")
        # print(f"PSNR: {psnr:.2f} dB")

        # plt.imshow(output)
        # plt.show()
        # cv2.imwrite("mediapiepe_Synthesis.png", cv2.cvtColor(output, cv2.COLOR_BGR2RGB)) 

        # plt.imshow(blended)
        # plt.show()
        # plt.imshow(warp2)
        # plt.show()
        # plt.imshow(warp1)
        # plt.show()
        # plt.imshow(headL)
        # plt.show()
        

    # -------------------
    # end of mediapipe rein
    # -------------------


    # disparityIL = warp_disparity_gpu(disparityLR, alpha, height, width)
    start = time.time()
    # remove ghosting
    if faceL is not None and faceR is not None:
        foreground_disparity = get_foreground_disparity(disparityLR)
        # plt.imshow(foreground_disparity)
        # plt.show()
        disparityIL_helper = np.ones((imgL.shape[0], imgL.shape[1], 1), np.float32) * -1
        for y in range(height):
            for x in range(width):
                if(foreground_disparity[y,x]):
                    disparity = foreground_disparity[y, x]
                    newX = math.floor(x - alpha*disparity)
                    newX = max(0, min(newX, width-1))
                    disparityIL_helper[y,newX] = disparity


        disparityIL_helper = fill_disparity_holes_gpu(disparityIL_helper, 20, 3)
        disparityIL_helper = np.round(disparityIL_helper)

        boundaryLR, diff = detect_EOBMR(disparityLR, 70, 15)
        # show_boundary(imgL_rgb, boundaryLR)
        # boundaryIL = detect_EOBMV(disparityIL_helper, 5)
        combined_boundary = cv2.bitwise_and(boundaryLR, (disparityIL_helper == -1).astype(np.uint8))

        disparityLR[combined_boundary == 1] = -1
    
    start = time.time()
    disparityIL = generate_intermediate_disparity(disparityLR, alpha, -1)
    # disparityIL = forward_warping(disparityLR, alpha, width, height, True)
    print("time taken for complex forward warping " + str(time.time() - start))
    print("\n")

    # for y in range(height):
    #     for x in range(width):
    #         if disparityLR[y, x]:
    #             disparity = disparityLR[y, x]
    #             newX = math.floor(x - alpha*disparity)
    #             # newX = max(0, min(newX, width-1))
    #             if newX < 0 or newX >= width: continue
    #             if(disparityIL[y,newX] == -1 or disparityIL[y,newX] < disparity):
    #                 disparityIL[y,newX] = disparity
    # print("Intermediate disparity " + str(time.time() - start))

    # Fill holes
    start = time.time()
    disparityIL = fill_disparity_holes_gpu(disparityIL, 20, 3)
    # disparityIL_helper = np.round(disparityIL)
    # print("time taken for complex crack filling " + str(time.time() - start))
    # print("\n")
    
    # combined_boundary = cv2.bitwise_or(boundaryLR, combined_boundary)

    # plt.subplot(1, 2, 1)  # (rows, columns, index) 
    # plt.imshow(disparityLR)
    # plt.title('Left Image')
    # plt.axis('off')

    # # Display imgR
    # plt.subplot(1, 2, 2)
    # plt.imshow(disparityIL_helper)
    # plt.title('Right Image')
    # plt.axis('off')
    # plt.show()

    # visualize_boundaries(imgL_rgb, boundaryLR, "Boundary LR")
    # visualize_boundaries(imgL_rgb, boundaryIL, "Boundary IL")
    # visualize_boundaries(imgL_rgb, combined_boundary, "Combined boundary")
    # visualize_combined_boundaries(imgL_rgb, boundaryLR, boundaryIL)

    # smooth the edges
    disparityIL = blur_boundaries(disparityIL)
    disparityIL = clean_disparity_map(disparityIL)
    # for y in range(height):
    #     for x in range(1,width-1):
    #         if ((np.std([disparityIL[y,x-1], disparityIL[y,x], disparityIL[y,x+1]]) > 20 or disparityIL[y,x] == -1) and np.std([disparityIL[y,x-1], disparityIL[y,x+1]]) < 3):
    #             disparityIL[y,x] = np.average([disparityIL[y,x-1], disparityIL[y,x+1]])

    # start = time.time()
    # disparityIL = simple_crack_filling(disparityIL)
    # print("time taken for crack filling " + str(time.time() - start))
    # print("\n")

    # disparityIL[combined_boundary == 1] = -1
    # visualize_boundaries(imgIL, boundaryIL, "Boundary IL")
    # visualize_boundaries(imgIL, combined_boundary, "Combined Boundaries")

    # for y in range(height):
    #     for x in range(width):
    #         if (disparityIL[y, x] >= 0):
    #             if (x + int(np.round(alpha*disparityIL[y, x]))) < width:
    #                 imgIL[y, x] = imgL_rgb[y, x + int(np.round(alpha*disparityIL[y, x]))]
    
    start = time.time()
    imgIL = warp_image_cv2(imgL_rgb, disparityIL, 1, alpha)
    # imgIL = backward_warping(imgL_rgb, disparityIL, 1, alpha)
    # print("time taken for complex backward wraping " + str(time.time() - start))
    # print("\n")


    if faceL is not None and faceR is not None:
        foreground_disparity = get_foreground_disparity(disparityRL)
        disparityIR_helper = np.ones((imgL.shape[0], imgL.shape[1], 1), np.float32) * -1
        for y in range(height):
            for x in range(width):
                if(foreground_disparity[y,x]):
                    disparity = foreground_disparity[y, x]
                    newX = math.floor(x + (1-alpha)*disparity)
                    newX = max(0, min(newX, width-1))
                    disparityIR_helper[y,newX] = disparity

        disparityIR_helper = fill_disparity_holes_gpu(disparityIR_helper, 20, 3)
        disparityIR_helper = np.round(disparityIR_helper)

        # remove ghosting
        boundaryRL, diff = detect_EOBMR(disparityRL, 70, 15)
        # boundaryIR = detect_EOBMV(disparityIR_helper, 5)
        combined_boundary = cv2.bitwise_and(boundaryRL, (disparityIR_helper == -1).astype(np.uint8))

        disparityRL[combined_boundary == 1] = -1

    disparityIR = generate_intermediate_disparity(disparityRL, (1-alpha), 1)
    # disparityIR = forward_warping(disparityRL, alpha, width, height, False)
    
    # for y in range(height):
    #     for x in range(width-1, -1, -1):
    #         if disparityRL[y, x]:
    #             disparity = disparityRL[y, x]
    #             newX = math.floor(x + (1-alpha)*disparity)
    #             newX = max(0, min(newX, width-1))
    #             disparityIR[y,newX] = disparity

    # Fill holes
    start = time.time()
    disparityIR = fill_disparity_holes_gpu(disparityIR, 20, 3)
    
    
    

    # smooth the edges
    disparityIR = blur_boundaries(disparityIR)
    disparityIR = clean_disparity_map(disparityIR)
    # for y in range(height):
    #     for x in range(1,width-1):
    #         if ((np.std([disparityIR[y,x-1], disparityIR[y,x], disparityIR[y,x+1]]) > 20 or disparityIR[y,x] == -1) and np.std([disparityIR[y,x-1], disparityIR[y,x+1]]) < 3):
    #             disparityIR[y,x] = np.average([disparityIR[y,x-1], disparityIR[y,x+1]])

    # disparityIR = simple_crack_filling(disparityIR)
    # print("cracks filling " + str(time.time() - start))
    # plt.imshow(disparityIL)
    # plt.show()
    # plt.imshow(disparityIR)
    # plt.show()

    imgIR = warp_image_cv2(imgR_rgb, disparityIR, -1, (1-alpha))
    # disparityIR[combined_boundary == 1] = -1
    # plt.imshow(imgIR)
    # plt.show()
    start = time.time()
    # for y in range(height):
    #     for x in range(width):
    #         if (disparityIR[y, x] >= 0):
    #             if(x - int(np.round((1-alpha)*disparityIR[y, x]))) > -1:
    #                 imgIR[y, x] = imgR_rgb[y, x - int(np.round((1-alpha)*disparityIR[y, x]))]


    # imgIR = backward_warping(imgR_rgb, disparityIR, -1, (1-alpha))
    # print("imgIR " + str(time.time() - start))

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

    # remove ghosting
    # boundaryLR, diff = detect_EOBMR(disparityLR, 70, 5)
    # boundaryIL = detect_EOBMV(disparityIL, 5)
    # combined_boundary = cv2.bitwise_and(boundaryLR, boundaryIL)

    # disparityIL[combined_boundary == 1] = -1

    # boundaryRL, diff = detect_EOBMR(disparityRL, 70, 5)
    # boundaryIR = detect_EOBMV(disparityIR, 5)
    # combined_boundary = cv2.bitwise_and(boundaryRL, boundaryIR)

    # disparityIR[combined_boundary == 1] = -1
    
    # cv2.imwrite("imgIL.png", cv2.cvtColor(imgIL, cv2.COLOR_BGR2RGB))
    # cv2.imwrite("imgIR.png", cv2.cvtColor(imgIR, cv2.COLOR_BGR2RGB))

    check_pixels = np.zeros((imgL.shape[0], imgL.shape[1], 1), np.uint8)

    # disparity_normalized = (disparityIL - disparityIL.min()) / (disparityIL.max() - disparityIL.min())
    # # Scale to 0-255 and convert to uint8
    # disparity_uint8 = (disparity_normalized * 255).astype(np.uint8)
    # # cv2.imwrite("disparityIL.png", disparity_uint8)

    # disparity_normalized = (disparityIR - disparityIR.min()) / (disparityIR.max() - disparityIR.min())
    # # Scale to 0-255 and convert to uint8
    # disparity_uint8 = (disparity_normalized * 255).astype(np.uint8)
    # cv2.imwrite("disparityIR.png", disparity_uint8)

    # start = time.time()
    # imgI = merge_views_simple(imgIL, disparityIL, imgIR, disparityIR, alpha)
    imgI = merge_disparity_views(imgIL, imgIR, disparityIL, disparityIR, alpha)
    # print("time taken for complex merging " + str(time.time() - start))
    # print("\n")

    # for y in range(height):
    #     for x in range(width):
    #         if (not np.array_equal(disparityIL[y, x], [-1]) and not np.array_equal(disparityIR[y, x], [-1])):
    #             if(not np.array_equal(disparityIL[y, x], [0]) and not np.array_equal(disparityIR[y, x], [0]) and abs(disparityIL[y, x] - disparityIR[y, x]) < 20):
    #                 # diff = np.mean(np.abs(imgIL[y, x] - imgIR[y, x]))
    #                 # if(diff > 250):
    #                 #     imgI[y, x] = [0,0, 255]
    #                 # else:
    #                 #     interpolated = alpha * imgIL[y, x] + (1 - alpha) * imgIR[y, x]
    #                 #     imgI[y, x] = interpolated.astype(np.uint8)
    #                 # imgI[y, x] = alpha*imgIL[y, x] + (1-alpha)*imgIR[y, x]
    #                 if(disparityIL[y, x]*(1-alpha) > (alpha)*disparityIR[y, x]):
    #                     imgI[y, x] = imgIL[y, x]
    #                 else:
    #                     imgI[y, x] = imgIR[y, x]
    #             elif (abs(disparityIL[y, x]) > abs(disparityIR[y, x])):
    #                 imgI[y, x] = imgIL[y, x]
    #             elif (np.array_equal(imgIR[y, x], [0, 0, 0]) and not np.array_equal(imgIL[y, x], [0, 0, 0])): # in some case bot have 0 disparities but imgIR also has black pixels
    #                 imgI[y, x] = imgIL[y, x]
    #             else:
    #                 imgI[y, x] = imgIR[y, x]
    #         elif (not np.array_equal(disparityIL[y, x], [-1])):
    #             imgI[y, x] = imgIL[y, x]
    #         elif (not np.array_equal(disparityIR[y, x], [-1])):
    #             imgI[y, x] = imgIR[y, x]

    # print("Merging " + str(time.time() - start))
    # return imgI, disparityIL, disparityIR, imgIL, imgIR
    # start = time.time()
    if faceL is not None and faceR is not None and not top_down_imgs:

        # imgI[blended_mask > 0] = (0.5 * blended[blended_mask > 0] + 0.5 * imgI[blended_mask > 0]).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # Adjust size as needed
        inner_face_mask = cv2.erode(blended_mask.astype(np.uint8), kernel, iterations=1)
        edges_mask = blended_mask.astype(np.uint8) - inner_face_mask
        # visualize_boundaries(blended, edges_mask)

        blended_mask_uint8 = blended_mask.astype(np.uint8) * 255
        disparity_exists_mask = (disparityIL_helper[:, :, 0] != -1).astype(np.uint8) * 255
        if top_down_imgs:
            blended_mask = blended_mask.astype(np.uint8) * 255
            blended_mask = cv2.rotate(blended_mask, cv2.ROTATE_90_CLOCKWISE)
            blended = cv2.rotate(blended, cv2.ROTATE_90_CLOCKWISE)
            leftover = ((blended_mask > 0) & (disparityIL_helper[:, :, 0] == -1)).astype(np.uint8) * 255
            leftover = (leftover & (disparityIR_helper[:, :, 0] == -1)).astype(np.uint8) * 255
            # plt.imshow(leftover)
            # plt.show()
        else:
            leftover = (blended_mask & (disparityIL_helper[:, :, 0] == -1)).astype(np.uint8) * 255
            # plt.imshow(leftover)
            # plt.show()
            leftover = (leftover & (disparityIR_helper[:, :, 0] == -1)).astype(np.uint8) * 255
            # plt.imshow(leftover)
            # plt.show()

        # plt.imshow(blended)
        # plt.show()

        # leftover: tvoje pôvodné 255/0 binárne maska v uint8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(leftover, connectivity=8)

        clean_leftover = np.zeros_like(leftover)
        min_blob_size = 10   # doladiť podľa rozlíšenia, napr. 200–1000 px

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_blob_size:
                clean_leftover[labels == label] = 255
        leftover = clean_leftover

        # plt.imshow(leftover)
        # plt.show()

        # kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        # leftover = cv2.morphologyEx(leftover, cv2.MORPH_CLOSE, kernel_close)

        # kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        # leftover = cv2.morphologyEx(leftover, cv2.MORPH_OPEN, kernel_open)

        # plt.imshow(leftover)
        # plt.show()

        imgI[leftover > 0] = blended[leftover > 0]
    # print("Time for media pipe face " + str(time.time() - start))
    return imgI, disparityIL, disparityIR, imgIL, imgIR


def create_edge_mask(disparity_map, mask_width=3):

    sobelx = cv2.Sobel(disparity_map, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(disparity_map, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    threshold = np.mean(magnitude) + np.std(magnitude)
    boundaries = (magnitude > threshold).astype(np.uint8) * 255

    # plt.imshow(boundaries)
    # plt.show()

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
    # EOBMR = cv2.dilate(EOBMR, np.ones((5,5), np.uint8))
    
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
    # Additional dilation to make boundaries thicker
    EOBMV = cv2.dilate(EOBMV, np.ones((5,5), np.uint8))
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

def merge_views_simple(imgIL, disparityIL, imgIR, disparityIR, alpha):
    imgI = np.zeros_like(imgIL)
    height, width = imgIL.shape[:2]
    for y in range(height):
        for x in range(width):
            dL = disparityIL[y, x]
            dR = disparityIR[y, x]
            pixL = imgIL[y, x]
            pixR = imgIR[y, x]
            if dL == -1 and dR == -1:
                    continue  # invalid pixel in both views
            
            elif dL != -1 and dR != -1:
                if abs(dL - dR) < 20:
                    imgI[y, x] = (1 - alpha) * pixL + alpha * pixR  # interpolate
                elif dL > dR:
                    imgI[y, x] = pixL  # closer object from left
                else:
                    imgI[y, x] = pixR  # closer object from right
        
            elif dL != -1:
                imgI[y, x] = pixL
        
            elif dR != -1:
                imgI[y, x] = pixR

    return imgI


def merge_disparity_views(imgIL, imgIR, disparityIL, disparityIR, alpha):
    disparityIL = disparityIL.squeeze()
    disparityIR = disparityIR.squeeze()
    
    # Basic masks
    invalid_both = ((disparityIL == -1) & (disparityIR == -1)).astype(np.uint8) * 255
    valid_both = (disparityIL != -1) & (disparityIR != -1)
    non_zero_both = (disparityIL != 0) & (disparityIR != 0)
    small_diff = np.abs(disparityIL - disparityIR) < 20
    
    # Black pixel detection
    black_IR = np.all(imgIR == 0, axis=2)
    black_IL = np.all(imgIL == 0, axis=2)
    
    imgI = np.zeros_like(imgIL)
    
    # Interpolation mask for similar disparities
    interpolate_mask = np.expand_dims(valid_both & non_zero_both & small_diff, axis=2)
    
    # For similar disparities, interpolate pixel values
    imgI = np.where(interpolate_mask, 
                   ((1-alpha) * imgIL + (alpha) * imgIR).astype(np.uint8),
                   imgI)
    
    # Handle remaining cases
    remaining_mask = np.expand_dims(valid_both & ~(non_zero_both & small_diff), axis=2)
    larger_disparity = np.expand_dims(np.abs(disparityIL) > np.abs(disparityIR), axis=2)
    black_pixels = np.expand_dims(black_IR & ~black_IL, axis=2)
    
    imgI = np.where(remaining_mask & larger_disparity, imgIL, imgI)
    imgI = np.where(remaining_mask & black_pixels, imgIL, imgI)
    imgI = np.where(remaining_mask & ~larger_disparity & ~black_pixels, imgIR, imgI)
    
    # Single valid cases
    imgI = np.where(np.expand_dims(disparityIL != -1, axis=2) & 
                   np.expand_dims(disparityIR == -1, axis=2), imgIL, imgI)
    imgI = np.where(np.expand_dims(disparityIR != -1, axis=2) & 
                   np.expand_dims(disparityIL == -1, axis=2), imgIR, imgI)
    
    # === 2. Pixel-wise color consistency check (choose visually closest pixel)
    small_disp_diff = np.abs(disparityIL - disparityIR) < 20
    diff = np.linalg.norm(imgIL.astype(np.float32) - imgIR.astype(np.float32), axis=2)
    good_fit_mask = valid_both & ~small_disp_diff & (diff < 30)  # adjust threshold if needed
    prefer_left_mask = good_fit_mask & (disparityIL > disparityIR)
    prefer_right_mask = good_fit_mask & ~prefer_left_mask

    # plt.imshow(diff)
    # plt.show()

    imgI = np.where(np.expand_dims(prefer_left_mask, 2), imgIL, imgI)
    imgI = np.where(np.expand_dims(prefer_right_mask, 2), imgIR, imgI)
    
    imgI = cv2.inpaint(imgI, invalid_both, 3, cv2.INPAINT_TELEA)
    return imgI

def blur_boundaries(disparity_map):
    original_shape = disparity_map.shape
    was_3d = False

    # Ak je 3D mapa s jedným kanálom (H, W, 1), konvertuj na 2D
    if disparity_map.ndim == 3 and disparity_map.shape[2] == 1:
        disparity_map = disparity_map.squeeze(axis=2)
        was_3d = True

    # Vytvor masku platných hodnôt
    valid_mask = (disparity_map != -1).astype(np.uint8)

    # Nahradenie -1 nulami pre spracovanie
    temp_disparity = np.where(disparity_map == -1, 0, disparity_map).astype(np.float32)

    # Median blur
    blurred_disparity = cv2.medianBlur(temp_disparity, 5)

    # Vrátenie -1 na neplatné pozície
    smoothed_disparity = np.where(valid_mask == 1, blurred_disparity, -1)

    # Ak bola pôvodne 3D, pridaj späť dimenziu
    if was_3d:
        smoothed_disparity = smoothed_disparity[..., np.newaxis]

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


def visualize_boundaries(image, boundary_mask, title="Boundary Visualization"):
    """
    Overlay boundary mask on the original image for visualization.
    
    Args:
        image: Original image (grayscale or color)
        boundary_mask: Binary boundary mask (should be single-channel)
        title: Title of the visualization
    """
    # Convert grayscale image to color if needed
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    # Convert mask to 3 channels for color overlay
    boundary_mask_color = cv2.cvtColor(boundary_mask * 255, cv2.COLOR_GRAY2BGR)

    # Color the boundaries red (modify the R channel)
    boundary_overlay = image_color.copy()
    boundary_overlay[:, :, 2] = np.where(boundary_mask == 1, 255, boundary_overlay[:, :, 2])  # Red channel

    # Blend original and boundary overlay
    blended = cv2.addWeighted(image_color, 0.7, boundary_overlay, 0.3, 0)
    # blended = cv2.rotate(blended, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Show the result
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
    plt.title(title)
    plt.axis("off")
    plt.show()



def visualize_combined_boundaries(image, boundary1, boundary2, title="Combined Boundaries"):
    """
    Overlay two boundary masks on the original image in different colors.
    
    Args:
        image: Original image (grayscale or color)
        boundary1: First boundary mask (should be single-channel)
        boundary2: Second boundary mask (should be single-channel)
        title: Title of the visualization
    """
    # Convert grayscale image to color if needed
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    # Create an empty color mask
    boundary_overlay = np.zeros_like(image_color)

    # Assign colors to boundaries
    boundary_overlay[:, :, 2] = np.where(boundary1 == 1, 255, 0)  # Red channel for boundary1 (boundaryRL)
    boundary_overlay[:, :, 0] = np.where(boundary2 == 1, 255, 0)  # Blue channel for boundary2 (boundaryIR)

    # Overlapping areas become purple (red + blue)
    overlap_mask = (boundary1 == 1) & (boundary2 == 1)
    boundary_overlay[overlap_mask] = [255, 0, 255]  # Purple (Red + Blue)

    # Blend the overlay with the original image
    blended = cv2.addWeighted(image_color, 0.7, boundary_overlay, 0.3, 0)
    # blended = cv2.rotate(blended, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Show the result
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
    plt.title(title)
    plt.axis("off")
    plt.show()