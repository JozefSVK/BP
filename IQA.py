import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import pandas as pd
from tqdm import tqdm

def compare_images(gt_folder, result_folder, gt_extension=".jpeg", result_extension=".png", resize_factor=1/3, save_plots=True, plots_folder="comparison_plots"):
    """
    Compare all ground truth images with corresponding result images.
    
    Args:
        gt_folder (str): Path to ground truth images folder
        result_folder (str): Path to result images folder
        gt_extension (str): Extension of ground truth images
        result_extension (str): Extension of result images
        resize_factor (float): Resize factor for ground truth images (set to 1 if no resize needed)
        save_plots (bool): Whether to save plots instead of displaying them
        plots_folder (str): Folder to save plots if save_plots is True
    
    Returns:
        pd.DataFrame: DataFrame with comparison metrics for all image pairs
    """
    # Create plots folder if saving plots
    if save_plots and not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    
    # Get all ground truth files
    gt_files = [f for f in os.listdir(gt_folder) if f.endswith(gt_extension)]
    
    # Initialize results list
    results = []
    
    # Process each ground truth file
    for gt_file in tqdm(gt_files, desc="Comparing images"):
        # Get base filename without extension
        base_name = os.path.splitext(gt_file)[0]
        
        # Construct the corresponding result filename
        result_file = base_name + result_extension
        result_path = os.path.join(result_folder, result_file)
        
        # Skip if result file doesn't exist
        if not os.path.exists(result_path):
            print(f"Warning: No matching result file for {gt_file}")
            continue
        
        # Load images
        gt_path = os.path.join(gt_folder, gt_file)
        ground_truth = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        
        # Resize ground truth if needed
        if resize_factor != 1:
            ground_truth = cv2.resize(ground_truth, None, fx=resize_factor, fy=resize_factor)
        
        distorted_image = cv2.imread(result_path, cv2.IMREAD_COLOR)
        
        # Ensure images are the same size
        if ground_truth.shape != distorted_image.shape:
            print(f"Warning: Size mismatch for {gt_file}. Resizing result image to match ground truth.")
            distorted_image = cv2.resize(distorted_image, (ground_truth.shape[1], ground_truth.shape[0]))
        
        # Convert to grayscale
        ground_truth_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
        distorted_gray = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2GRAY)
        
        # Compute absolute difference (Error map)
        diff = cv2.absdiff(ground_truth, distorted_image)
        diff_gray = cv2.absdiff(ground_truth_gray, distorted_gray)
        
        # Compute SSIM difference map
        ssim_gray_value, ssim_map = ssim(ground_truth_gray, distorted_gray, full=True, data_range=255)
        
        # Compute MSE for color and grayscale images
        mse_value = mean_squared_error(ground_truth, distorted_image)
        mse_gray_value = mean_squared_error(ground_truth_gray, distorted_gray)
        
        # Compute PSNR
        psnr_value = psnr(ground_truth, distorted_image, data_range=255)
        psnr_gray_value = psnr(ground_truth_gray, distorted_gray, data_range=255)
        
        # Compute SSIM
        ssim_value = ssim(ground_truth, distorted_image, channel_axis=-1, data_range=255)
        
        # Store results
        results.append({
            'filename': base_name,
            'MSE_Color': mse_value,
            'MSE_Gray': mse_gray_value,
            'PSNR_Color': psnr_value,
            'PSNR_Gray': psnr_gray_value,
            'SSIM_Color': ssim_value,
            'SSIM_Gray': ssim_gray_value
        })
        
        # Create and display/save plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Ground Truth Image")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Result Image")
        axes[0, 1].axis("off")
        
        axes[0, 2].imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Absolute Difference (Color)")
        axes[0, 2].axis("off")
        
        axes[1, 0].imshow(ground_truth_gray, cmap="gray")
        axes[1, 0].set_title("Ground Truth (Grayscale)")
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(diff_gray, cmap="hot")
        axes[1, 1].set_title("Absolute Difference (Grayscale)")
        axes[1, 1].axis("off")
        
        axes[1, 2].imshow(ssim_map, cmap="coolwarm")
        axes[1, 2].set_title(f"SSIM Map (SSIM: {ssim_gray_value:.4f})")
        axes[1, 2].axis("off")
        
        plt.suptitle(f"Image Comparison: {base_name}")
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(plots_folder, f"{base_name}_comparison.png"), dpi=150)
            plt.close()
        else:
            plt.show()
    
    # Create DataFrame with results
    df_results = pd.DataFrame(results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(df_results.describe())
    
    # Optionally save results to CSV
    df_results.to_csv("comparison_results.csv", index=False)
    
    return df_results


if __name__ == "__main__":
    gt_folder = "dataset/dataset/GT"
    result_folder = "dataset/dataset/res"
    
    # Call the comparison function
    results = compare_images(
        gt_folder=gt_folder,
        result_folder=result_folder,
        gt_extension=".jpeg",
        result_extension=".png",
        resize_factor=1/3,
        save_plots=True,
        plots_folder="comparison_plots"
    )
    
    # Print the top 5 best and worst images based on SSIM
    print("\nTop 5 Best Images (by SSIM):")
    print(results.sort_values("SSIM_Color", ascending=False).head(5)[["filename", "SSIM_Color", "PSNR_Color"]])
    
    print("\nTop 5 Worst Images (by SSIM):")
    print(results.sort_values("SSIM_Color").head(5)[["filename", "SSIM_Color", "PSNR_Color"]])

    # Load images
    # ground_truth = cv2.imread("dataset/dataset/GT/5_5.jpeg", cv2.IMREAD_COLOR)
    # ground_truth = cv2.resize(ground_truth, None, fx=1/3, fy=1/3)
    # distorted_image = cv2.imread("dataset/custom_res/Middle_view.png", cv2.IMREAD_COLOR)
    # distorted_image2 = cv2.imread("dataset/dataset/res/5_5.png", cv2.IMREAD_COLOR)

    # ssim_value = ssim(ground_truth, distorted_image, channel_axis=-1, data_range=255)

    # print(f"SSIM (Color): {ssim_value:.4f}")

    # ssim_value = ssim(ground_truth, distorted_image2, channel_axis=-1, data_range=255)

    # print(f"SSIM (Color): {ssim_value:.4f}")



