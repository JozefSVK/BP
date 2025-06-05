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
import utilities
import argparse
import sys
import pandas as pd
import os
from skimage.metrics import structural_similarity as ssim

count = 1

# Kontrolná funkcia, ktorá overí cestu k súborom a výstupný priečinok
def check_paths_and_create_output(left_path, right_path, output_path):
    errors = []

    # Overenie vstupných obrázkov
    if not os.path.isfile(left_path):
        errors.append(f"Vstupný obrázok neexistuje: {left_path}")
    if not os.path.isfile(right_path):
        errors.append(f"Vstupný obrázok neexistuje: {right_path}")

    # Overenie výstupnej cesty – musí obsahovať aj názov súboru s príponou
    output_dir = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)

    # Musí mať príponu
    valid_extensions = ['.png', '.jpg', '.jpeg']
    if not any(output_filename.endswith(ext) for ext in valid_extensions):
        errors.append("Výstupná cesta musí obsahovať názov súboru s príponou (.png, .jpg, .jpeg)")

    # Ak priečinok neexistuje, vytvor ho
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return errors

def compare_and_save_results(generated_img, ground_truth_img_path, excel_path="results.xlsx", metadata={}):
    # Load ground truth image
    ground_truth = cv2.imread(ground_truth_img_path)
    if ground_truth is None:
        print(f"Chyba: Súbor {ground_truth_img_path} sa nepodarilo načítať.")
        return

    # Convert generated image to RGB
    generated_img = cv2.cvtColor(generated_img, cv2.COLOR_BGR2RGB)

    # Resize GT if needed
    if ground_truth.shape != generated_img.shape:
        ground_truth = cv2.resize(ground_truth, (generated_img.shape[1], generated_img.shape[0]))

    plt.imshow(generated_img)
    plt.show()
    plt.imshow(ground_truth)
    plt.show()

    # Valid pixel mask
    valid_mask = np.any(generated_img > 0, axis=2)

    # Extract valid pixels
    gen_pixels = generated_img[valid_mask]
    gt_pixels = ground_truth[valid_mask]

    # MAE, MSE, PSNR
    mae = np.mean(np.abs(gen_pixels.astype(np.float32) - gt_pixels.astype(np.float32)))
    mse = np.mean((gen_pixels.astype(np.float32) - gt_pixels.astype(np.float32)) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")

    # SSIM na valid maskovanom grayscale
    gray_gen = cv2.cvtColor(generated_img, cv2.COLOR_RGB2GRAY)
    gray_gt = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    gray_gen_valid = gray_gen[valid_mask]
    gray_gt_valid = gray_gt[valid_mask]

    if gray_gen_valid.size > 0:
        ssim_score = ssim(
            gray_gt_valid,
            gray_gen_valid,
            data_range=255
        )
    else:
        ssim_score = 0  # žiadne valid pixely

    # Record row
    row = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "PSNR": round(psnr, 2),
        "SSIM": round(ssim_score, 4),
        **metadata
    }

    # Append to Excel
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    df.to_excel(excel_path, index=False)
    print(f"Výsledky uložené do: {excel_path}")



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

def func(left_image_path, right_image_path, donwscale=0.5, model='IGEV', top_down_imgs = False, output_path=None):
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
    

    # call wrapper
    start = time.time()
    disparityLR, disparityRL = utilities.calculate_disparity(imgL, imgR, model)
    print("disparity calculator " + str(time.time() - start))

    height, width = imgL.shape[:2]
    print("Height ", height)
    print("Width", width)

    

    # plt.imshow(disparityLR)
    # plt.show()
    # save_disparity(disparityLR, 'background.png')
    filter_value = 120
    # disparityLR = filter_disparity_map(disparityLR, filter_value, None, 0)
    # disparityRL = filter_disparity_map(disparityRL, filter_value, None, 0)
    # save_disparity(disparityLR, 'disparityLR.png')
    # save_disparity(disparityRL, 'disparityRL.png')
    # plt.imshow(disparityRL)
    # plt.show()


    imgL_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    imgR_rgb = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

    # new_img = remove_pixels(imgL_rgb, disparityLR)
    # plt.imshow(new_img)
    # plt.show()


    alpha = 0.5
    

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


    imgI, disparityIL, disparityIR, imgIL, imgIR  = view_synthesis.create_intermediate_view(imgL, imgR, disparityLR, disparityRL, alpha, top_down_imgs)
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

    # compare_and_save_results(imgI, "dataset/1/2/0001.png", excel_path="2CameraV.xlsx",
    #                          metadata={
    #                             "Model": model,
    #                             "Alpha": alpha,
    #                             "Left": os.path.basename(left_image_path),
    #                             "Right": os.path.basename(right_image_path)
    #                          })

    # dispaly images
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
    if output_path is not None:
        cv2.imwrite(output_path, imgI)
        print(f"[INFO] Výsledný obrázok uložený do: {output_path}")
    else:
        cv2.imwrite("imgI.png", imgI)
        print(f"[INFO] Výstupný obrázok uložený ako: imgI.png (predvolený názov)")
    # imgIL = cv2.cvtColor(imgIL, cv2.COLOR_BGR2RGB)
    # imgIR = cv2.cvtColor(imgIR, cv2.COLOR_BGR2RGB)
    # global count
    # filename = f"new_res_head/H5_5.png"
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
    if(len(sys.argv) > 1):

        parser = argparse.ArgumentParser(description="Stereo image processing and view synthesis")
        parser.add_argument('--left', type=str, required=True, help='Path to left image')
        parser.add_argument('--right', type=str, required=True, help='Path to right image')
        parser.add_argument('--resize', type=float, default=1.0, help='Resize factor (e.g., 0.5)')
        parser.add_argument('--model', type=str, default='IGEV', help='Disparity model name')
        parser.add_argument('--topdown', action='store_true', help='Use if images are in top-down view')
        parser.add_argument('--output', type=str, required=True, help='Output path for the result image')

        args = parser.parse_args()

        # Overenie ciest a priečinka
        errors = check_paths_and_create_output(args.left, args.right, args.output)
        if errors:
            for error in errors:
                print(f"[CHYBA] {error}")
            sys.exit(1)

        func(
            left_image_path=args.left,
            right_image_path=args.right,
            donwscale=args.resize,
            model=args.model,
            top_down_imgs=args.topdown,
            output_path=args.output
        )

    else:
        # func('dataset/1/1/0003.png', 'dataset/1/1/0002.png')
        model_name = 'RAFT'
        top_down = True
        func('dataset/1/2/left.png', 'dataset/1/2/right.png', 1/3, model_name)

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
        # func('dataset/custom_dataset/bg/BL.jpeg', 'dataset/custom_dataset/bg/BR.jpeg', 1/3, model_name)
        # func('dataset/custom_dataset/bg/TL.jpeg', 'dataset/custom_dataset/bg/TR.jpeg', 1/3, model_name)
        # func('new_res_head/L5.png', 'new_res_head/R5.png', 1, model_name)
        # func('new_res_head/B5.png', 'new_res_head/T5.png', 1, model_name, top_down)

        # func('newestResult/L5.png', 'newestResult/R5.png', 1, model_name)
        # func('newestResult/B5.png', 'newestResult/T5.png', 1, model_name, top_down)

        # func('dataset/middlebury/view1.png', 'dataset/middlebury/view5.png', 1, model_name)
        # func('dataset/middlebury/im0.png', 'dataset/middlebury/im1.png', 1, model_name)