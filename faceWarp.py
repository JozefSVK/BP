import cv2
import mediapipe as mp
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh

# --- Landmark Detection ---
def detect_face_landmarks(image):
    """Detects face landmarks using MediaPipe Face Mesh."""
    if image is None:
        print("Error: Input image is None.")
        return None
    if image.size == 0:
        print("Error: Input image is empty.")
        return None

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print("Warning: No face detected in an image.")
            return None

        face_landmarks = results.multi_face_landmarks[0]
        img_h, img_w, _ = image.shape
        landmarks_px = np.array([(lm.x * img_w, lm.y * img_h)
                                for lm in face_landmarks.landmark], dtype=np.float32)
        return landmarks_px

# --- Degeneracy Check ---
def triangle_area(p1, p2, p3):
    """Calculates twice the signed area of the triangle. Close to zero means collinear."""
    val = p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
    return val

# --- ENHANCED Warping Function ---
def warp_image_piecewise_affine(src_img, src_points, dst_points, triangles, dst_shape):
    """
    Warps an image using piecewise affine transformation based on triangle meshes.
    Enhanced version with improved blending and edge handling.
    """
    dst_h, dst_w = dst_shape
    warped_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    # Using float32 for the mask to allow anti-aliasing and smoother blending
    warped_mask = np.zeros((dst_h, dst_w), dtype=np.float32)

    num_processed_tris = 0
    num_skipped_index = 0
    num_skipped_degenerate = 0
    num_skipped_affine_error = 0
    num_skipped_bbox = 0
    DEGENERACY_THRESHOLD = 1e-4

    if triangles is None:
        print("Error: No triangles provided for warping.")
        return warped_img, warped_mask
    if src_points is None or dst_points is None:
         print("Error: Source or destination points are None.")
         return warped_img, warped_mask
    if not np.all(np.isfinite(src_points)) or not np.all(np.isfinite(dst_points)):
        print("!!! WARNING: Source or Destination points array contain NaN or Inf values before loop!")

    # Create a temporary holder for all warped triangles to allow for better blending
    warped_triangles = []

    for i, tri_indices in enumerate(triangles):
        # Get vertices
        try:
            tri_indices_list = [int(idx) for idx in tri_indices]
            if any(idx >= len(src_points) or idx >= len(dst_points) or idx < 0 for idx in tri_indices_list):
                num_skipped_index += 1
                continue
            src_tri = src_points[tri_indices_list].astype(np.float32)
            dst_tri = dst_points[tri_indices_list].astype(np.float32)
        except (IndexError, ValueError):
            num_skipped_index += 1
            continue

        # Check for NaN/Inf
        if not np.all(np.isfinite(src_tri)) or not np.all(np.isfinite(dst_tri)):
            num_skipped_affine_error += 1
            continue

        # Check for Degenerate Triangles
        if src_tri.shape != (3, 2) or dst_tri.shape != (3, 2):
             num_skipped_affine_error += 1
             continue
        src_area = triangle_area(src_tri[0], src_tri[1], src_tri[2])
        dst_area = triangle_area(dst_tri[0], dst_tri[1], dst_tri[2])
        if abs(src_area) < DEGENERACY_THRESHOLD or abs(dst_area) < DEGENERACY_THRESHOLD:
            num_skipped_degenerate += 1
            continue

        # Calculate Affine Transform
        try:
            M = cv2.getAffineTransform(src_tri, dst_tri)
            if M is None:
                num_skipped_affine_error += 1
                continue
        except cv2.error as e:
            num_skipped_affine_error += 1
            continue

        # Bounding Box and Clipping
        try:
            x, y, w, h = cv2.boundingRect(dst_tri.astype(np.int32))
        except cv2.error as e:
            num_skipped_bbox += 1
            continue
        x, y, w, h = int(x), int(y), int(w), int(h)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(dst_w, x + w), min(dst_h, y + h)
        w_clip, h_clip = x2 - x1, y2 - y1
        if w_clip <= 0 or h_clip <= 0:
             num_skipped_bbox += 1
             continue

        # Warping Logic
        try:
            # Create high-res mask for better anti-aliasing
            bbox_mask = np.zeros((h_clip, w_clip), dtype=np.float32)
            dst_tri_rel = dst_tri - np.array([x1, y1], dtype=np.float32)
            
            # Use anti-aliasing for smoother edges
            cv2.fillConvexPoly(bbox_mask, dst_tri_rel.astype(np.int32), 1.0, cv2.LINE_AA, 0)
            
            # Apply a small Gaussian blur to the mask to further smooth edges
            bbox_mask = cv2.GaussianBlur(bbox_mask, (3, 3), 0.5)
            
            # Warp the full image
            img_warped_full = cv2.warpAffine(src_img, M, (dst_w, dst_h), 
                                           flags=cv2.INTER_LINEAR, 
                                           borderMode=cv2.BORDER_REPLICATE)
            
            img_warped_patch = img_warped_full[y1:y2, x1:x2]
            
            # Store this triangle's contribution
            warped_triangles.append((
                img_warped_patch, 
                bbox_mask,
                (y1, y2, x1, x2)
            ))
            
        except Exception as e:
            print(f"!!! ERROR during warping/masking for triangle {tri_indices}: {e}")
            continue

        num_processed_tris += 1

    # Now blend all triangles with better edge handling
    for img_patch, mask_patch, (y1, y2, x1, x2) in warped_triangles:
        try:
            # Extract current region
            current_region = warped_img[y1:y2, x1:x2].astype(np.float32)
            current_mask = warped_mask[y1:y2, x1:x2]
            
            # Calculate new weights
            new_weight = mask_patch
            old_weight = current_mask
            total_weight = old_weight + new_weight
            
            # Avoid division by zero
            valid_pixels = total_weight > 0
            
            # Initialize blend arrays
            blended_region = current_region.copy()
            blended_mask = current_mask.copy()
            
            # Only blend valid pixels
            if np.any(valid_pixels):
                # Normalize weights for valid pixels
                norm_old = np.zeros_like(old_weight)
                norm_new = np.zeros_like(new_weight)
                
                norm_old[valid_pixels] = old_weight[valid_pixels] / total_weight[valid_pixels]
                norm_new[valid_pixels] = new_weight[valid_pixels] / total_weight[valid_pixels]
                
                # Blend colors
                for c in range(3):
                    blended_region[..., c] = (
                        current_region[..., c] * norm_old + 
                        img_patch[..., c] * norm_new
                    )
                
                # Update mask
                blended_mask = np.maximum(current_mask, mask_patch)
            
            # Update the full image and mask
            warped_img[y1:y2, x1:x2] = blended_region.astype(np.uint8)
            warped_mask[y1:y2, x1:x2] = blended_mask
            
        except Exception as e:
            print(f"!!! ERROR during final blending: {e}")
            continue

    print(f"DEBUG Summary (Warping): Processed={num_processed_tris}, Skipped(IndexError)={num_skipped_index}, "
          f"Skipped(Degenerate)={num_skipped_degenerate}, Skipped(AffineError)={num_skipped_affine_error}, "
          f"Skipped(BBox)={num_skipped_bbox}")
    
    # Convert mask back to uint8 for compatibility
    warped_mask_uint8 = (warped_mask * 255).astype(np.uint8)
    return warped_img, warped_mask_uint8

# --- IMPROVED Blending Function ---
def blend_images_masked_alpha(img1, mask1, img2, mask2):
    """
    Blends two images using their masks for alpha weighting.
    Improved version with Gaussian blur for smoother transitions.
    """
    # Convert masks to float
    mask1_f = (mask1 / 255.0).astype(np.float32) if mask1.max() > 1 else mask1.astype(np.float32)
    mask2_f = (mask2 / 255.0).astype(np.float32) if mask2.max() > 1 else mask2.astype(np.float32)
    
    # Apply a slight Gaussian blur to the masks for smoother transitions
    mask1_f = cv2.GaussianBlur(mask1_f, (5, 5), 1.0)
    mask2_f = cv2.GaussianBlur(mask2_f, (5, 5), 1.0)
    
    # Calculate weights
    total_alpha = mask1_f + mask2_f
    zero_alpha_mask = total_alpha < 1e-6
    total_alpha[zero_alpha_mask] = 1.0
    weight1 = mask1_f / total_alpha
    weight2 = mask2_f / total_alpha
    weight1[zero_alpha_mask] = 0.0
    weight2[zero_alpha_mask] = 0.0
    
    # Convert images to float for blending
    img1_f = img1.astype(np.float32)
    img2_f = img2.astype(np.float32)
    
    # Blend images
    blended_img_f = img1_f * weight1[:, :, np.newaxis] + img2_f * weight2[:, :, np.newaxis]
    
    # Create combined mask
    combined_mask = (~zero_alpha_mask).astype(np.uint8) * 255
    
    # Additional post-processing for smoother appearance
    blended_img = blended_img_f.astype(np.uint8)
    
    # Apply slight blur to the final image to reduce visible seams (optional)
    # blended_img = cv2.GaussianBlur(blended_img, (3, 3), 0.5)
    
    return blended_img, combined_mask

# --- POST PROCESSING Function (NEW) ---
def post_process_face(blended_face, mask):
    """Apply post-processing to remove visible mesh lines and enhance quality."""
    # Create a copy to work on
    processed = blended_face.copy()
    
    # Convert to float for processing
    processed_f = processed.astype(np.float32)
    
    # 1. Apply bilateral filter to smooth while preserving edges
    # processed = cv2.bilateralFilter(processed, 9, 75, 75)
    
    # 2. Apply a very slight Gaussian blur to further reduce mesh lines
    # processed = cv2.GaussianBlur(processed, (3, 3), 0.5)
    
    # 3. Apply mask
    if mask is not None:
        mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask
        processed = cv2.bitwise_and(processed, mask_3c)
    
    return processed

# --- Main Orchestration Function ---
def generate_intermediate_face(img_l_path, img_r_path):
    """
    Generates the intermediate face view using Delaunay triangulation.
    Enhanced version with smoother blending and post-processing.
    """
    print("-" * 20)
    print(f"Processing: L='{img_l_path}', R='{img_r_path}' (Using Enhanced Delaunay)")
    print("-" * 20)

    print("Loading images...")
    img_l = cv2.imread(img_l_path)
    if img_l is None or img_l.size == 0:
        print(f"Error: Could not load or empty image {img_l_path}")
        return None
    img_r = cv2.imread(img_r_path)
    if img_r is None or img_r.size == 0:
        print(f"Error: Could not load or empty image {img_r_path}")
        return None

    # --- 1. Detect Landmarks ---
    print("Detecting landmarks...")
    kp_l_face = detect_face_landmarks(img_l)
    if kp_l_face is None: return None
    print(f"  Detected {len(kp_l_face)} landmarks for left image.")
    kp_r_face = detect_face_landmarks(img_r)
    if kp_r_face is None: return None
    print(f"  Detected {len(kp_r_face)} landmarks for right image.")

    if len(kp_l_face) != len(kp_r_face) or len(kp_l_face) == 0:
        print("!!! ERROR: Landmark count mismatch or zero landmarks!")
        return None

    # --- 2. Calculate Intermediate Landmarks ---
    print("Calculating intermediate landmarks...")
    kp_intermediate_face = 0.5 * kp_l_face + 0.5 * kp_r_face
    if not np.all(np.isfinite(kp_intermediate_face)):
         print("!!! ERROR: Intermediate keypoints calculation resulted in NaN or Inf!")
         return None

    # --- 3. Calculate Delaunay Triangulation ---
    print("Calculating Delaunay triangulation on intermediate points...")
    triangles = None
    try:
        # Perform Delaunay triangulation on the *intermediate* points
        delaunay = Delaunay(kp_intermediate_face)
        # delaunay.simplices gives the list of triangles (indices of points)
        triangles = delaunay.simplices
        print(f"  Calculated {len(triangles)} triangles.")
    except Exception as e:
        print(f"!!! ERROR during Delaunay triangulation: {e}")
        print("!!! Check if landmarks are valid (e.g., not all collinear).")
        return None # Cannot proceed without triangles

    # --- 4. & 5. Warp Images Piecewise ---
    h, w = img_l.shape[:2]
    dst_shape = (h, w)
    print(f"Target warp dimensions: {dst_shape}")

    print("Warping left image...")
    warped_l_face, warped_mask_l = warp_image_piecewise_affine(
        img_l, kp_l_face, kp_intermediate_face, triangles, dst_shape
    )

    print("Warping right image...")
    warped_r_face, warped_mask_r = warp_image_piecewise_affine(
        img_r, kp_r_face, kp_intermediate_face, triangles, dst_shape
    )

    # --- 6. Blend Warped Faces ---
    print("Blending warped images...")
    blended_face, blended_mask = blend_images_masked_alpha(
        warped_l_face, warped_mask_l, warped_r_face, warped_mask_r
    )
    
    # --- 7. Apply Post-Processing (NEW STEP) ---
    print("Applying post-processing to remove mesh lines...")
    final_result = post_process_face(blended_face, blended_mask)
    
    print("-" * 20)
    print("Intermediate face generation complete.")
    print("-" * 20)
    return final_result, blended_mask, warped_l_face, warped_mask_l, warped_r_face, warped_mask_r

def compare_with_ground_truth(generated_face, ground_truth_path, blended_mask=None):
    """
    Compares the generated intermediate face with the ground truth image.
    Strictly focuses ONLY on the face region, excluding all background and black space.
    
    Args:
        generated_face: The generated intermediate face image
        ground_truth_path: Path to the ground truth image
        blended_mask: Optional mask that defines the face region
        
    Returns:
        Comparison metrics and visualization
    """
    print("-" * 20)
    print(f"Comparing with ground truth: '{ground_truth_path}'")
    
    # Load ground truth image
    ground_truth = cv2.imread(ground_truth_path)
    if ground_truth is None:
        print(f"Error: Could not load ground truth image {ground_truth_path}")
        return None
    
    # Ensure both images are in the same color space for comparison
    generated_face_bgr = cv2.cvtColor(generated_face, cv2.COLOR_RGB2BGR) if len(generated_face.shape) == 3 and generated_face.shape[2] == 3 else generated_face
    
    # Resize ground truth if dimensions don't match
    if ground_truth.shape[:2] != generated_face_bgr.shape[:2]:
        print(f"Resizing ground truth from {ground_truth.shape[:2]} to {generated_face_bgr.shape[:2]}")
        ground_truth = cv2.resize(ground_truth, (generated_face_bgr.shape[1], generated_face_bgr.shape[0]))
    
    # --- Create or refine face region mask ---
    face_mask = None
    
    # If a blended mask is provided, we'll use it as a starting point
    if blended_mask is not None and blended_mask.size > 0:
        # Ensure proper format for the mask
        if len(blended_mask.shape) == 3 and blended_mask.shape[2] == 3:
            face_mask = cv2.cvtColor(blended_mask, cv2.COLOR_BGR2GRAY)
        else:
            face_mask = blended_mask.copy()
            
        # Make sure the mask is binary
        _, face_mask = cv2.threshold(face_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours to remove small, disconnected regions
        contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Keep only the largest contour (the face)
            largest_contour = max(contours, key=cv2.contourArea)
            face_mask = np.zeros_like(face_mask)
            cv2.drawContours(face_mask, [largest_contour], 0, 255, -1)
    
    # If no mask is available yet, detect face in the ground truth
    if face_mask is None or np.sum(face_mask) == 0:
        print("Using MediaPipe to detect face region...")
        
        # Try the ground truth image first
        face_landmarks = detect_face_landmarks(ground_truth)
        
        # If that fails, try the generated face
        if face_landmarks is None:
            face_landmarks = detect_face_landmarks(generated_face_bgr)
            
        if face_landmarks is not None:
            # Create a convex hull around the face landmarks
            face_mask = np.zeros(ground_truth.shape[:2], dtype=np.uint8)
            hull_points = cv2.convexHull(face_landmarks.astype(np.int32))
            cv2.fillConvexPoly(face_mask, hull_points, 255)
            print(f"Face region detected with area of {np.sum(face_mask > 0) / (face_mask.shape[0] * face_mask.shape[1]) * 100:.1f}% of image")
        else:
            print("WARNING: Could not detect face in either image. Creating a central oval mask as fallback.")
            face_mask = np.zeros(ground_truth.shape[:2], dtype=np.uint8)
            center_x, center_y = face_mask.shape[1] // 2, face_mask.shape[0] // 2
            axes_length = (int(face_mask.shape[1] * 0.4), int(face_mask.shape[0] * 0.5))
            cv2.ellipse(face_mask, (center_x, center_y), axes_length, 0, 0, 360, 255, -1)
    
    # --- Remove any black space and background from the comparison ---
    
    # For both images, create non-black masks
    non_black_mask_gen = np.any(generated_face_bgr > 20, axis=2).astype(np.uint8) * 255
    non_black_mask_gt = np.any(ground_truth > 20, axis=2).astype(np.uint8) * 255
    
    # Combine all masks - we only want to compare pixels that:
    # 1. Are part of the face (per face_mask)
    # 2. Are not black in either image
    combined_mask = cv2.bitwise_and(face_mask, cv2.bitwise_and(non_black_mask_gen, non_black_mask_gt))
    
    # Apply erosion to remove boundary pixels that might be partially filled
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
    
    # Check if we have enough pixels to compare
    num_valid_pixels = np.sum(combined_mask > 0)
    total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
    print(f"Comparing {num_valid_pixels} pixels ({num_valid_pixels/total_pixels*100:.1f}% of image)")
    
    if num_valid_pixels < 100:  # Arbitrary threshold
        print("WARNING: Very few valid pixels for comparison. Results may be unreliable.")
    
    # Apply the refined mask to both images
    generated_face_masked = cv2.bitwise_and(generated_face_bgr, generated_face_bgr, mask=combined_mask)
    ground_truth_masked = cv2.bitwise_and(ground_truth, ground_truth, mask=combined_mask)
    
    # Convert to grayscale for SSIM comparison (but only using our valid pixels)
    generated_gray = cv2.cvtColor(generated_face_masked, cv2.COLOR_BGR2GRAY)
    ground_truth_gray = cv2.cvtColor(ground_truth_masked, cv2.COLOR_BGR2GRAY)
    
    # Calculate metrics only on the valid face region
    # For SSIM, we'll use the masked images
    ssim_score = ssim(generated_gray, ground_truth_gray, data_range=255)
    
    # For PSNR and MSE, we'll only consider pixels where the mask is non-zero
    mask_indices = combined_mask > 0
    if np.any(mask_indices):
        # Reshape to 1D arrays for faster computation
        gen_pixels = generated_face_bgr[mask_indices].astype(np.float32)
        gt_pixels = ground_truth[mask_indices].astype(np.float32)
        
        # Calculate MSE only on valid pixels
        mse = np.mean((gen_pixels - gt_pixels) ** 2)
        
        # Calculate PSNR from MSE
        if mse > 0:
            psnr_score = 10 * np.log10((255 ** 2) / mse)
        else:
            psnr_score = float('inf')  # Perfect match
    else:
        mse = float('nan')
        psnr_score = float('nan')
    
    # Create difference image (only where the mask is active)
    diff = cv2.absdiff(ground_truth_masked, generated_face_masked)
    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)  # Apply color map for better visualization
    
    # Convert images to RGB for display
    generated_face_rgb = cv2.cvtColor(generated_face_masked, cv2.COLOR_BGR2RGB)
    ground_truth_rgb = cv2.cvtColor(ground_truth_masked, cv2.COLOR_BGR2RGB)
    diff_rgb = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
    
    # Create outlines of the face region for visualization
    face_contours = np.zeros_like(ground_truth)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(face_contours, contours, -1, (0, 255, 0), 2)
    
    # Overlay contour on the original images for display
    ground_truth_outline = ground_truth.copy()
    generated_outline = generated_face_bgr.copy()
    alpha = 0.7  # Transparency factor
    ground_truth_outline = cv2.addWeighted(ground_truth_outline, 1, face_contours, 0.3, 0)
    generated_outline = cv2.addWeighted(generated_outline, 1, face_contours, 0.3, 0)
    
    # Convert to RGB for display
    ground_truth_outline_rgb = cv2.cvtColor(ground_truth_outline, cv2.COLOR_BGR2RGB)
    generated_outline_rgb = cv2.cvtColor(generated_outline, cv2.COLOR_BGR2RGB)
    
    print(f"Comparison Metrics (FACE REGION ONLY):")
    print(f"  SSIM: {ssim_score:.4f} (higher is better, 1.0 is perfect match)")
    print(f"  PSNR: {psnr_score:.2f} dB (higher is better)")
    print(f"  MSE: {mse:.2f} (lower is better, 0 is perfect match)")
    print("-" * 20)
    
    return {
        'ssim': ssim_score, 
        'psnr': psnr_score, 
        'mse': mse,
        'mask': combined_mask,
        'images': (generated_face_rgb, ground_truth_rgb, diff_rgb),
        'outline_images': (generated_outline_rgb, ground_truth_outline_rgb)
    }

# --- Example Usage ---
if __name__ == "__main__":
    # ==================================================
    # !!! IMPORTANT: SET YOUR IMAGE PATHS HERE !!!
    # ==================================================
    # left_image_path = 'dataset/dataset/BL.jpeg'   # e.g., 'dataset/1/1/0003.png'
    # right_image_path = 'dataset/dataset/BR.jpeg' # e.g., 'dataset/1/1/0002.png'
    # ground_truth_path = 'dataset/dataset/GT/B5.jpeg'
    left_image_path = 'dataset/1/1/0003.png'
    right_image_path = 'dataset/1/1/0002.png'
    ground_truth_path = 'dataset/1/1/0001.png'
    # ==================================================
    output_base = "intermediate_face_output" # Base name for output files

    result_data = generate_intermediate_face(left_image_path, right_image_path)

    if result_data:
        blended_face_result, blended_mask_result, warped_l, mask_l_uint8, warped_r, mask_r_uint8 = result_data

        # Ensure masks are uint8 for bitwise_and and saving
        mask_l_uint8 = mask_l_uint8.astype(np.uint8)
        mask_r_uint8 = mask_r_uint8.astype(np.uint8)
        blended_mask_result = blended_mask_result.astype(np.uint8)

        try:
            # Apply masks to warped images before saving for clarity
            warped_l_masked = cv2.bitwise_and(warped_l, warped_l, mask=mask_l_uint8)
            warped_r_masked = cv2.bitwise_and(warped_r, warped_r, mask=mask_r_uint8)
            
            # # Save the final blended face
            # output_path_blended = f"{output_base}_blended.png"
            # if cv2.imwrite(output_path_blended, blended_face_result):
            #     print(f"Blended intermediate face saved to: {output_path_blended}")
            # else:
            #     print(f"!!! Error saving blended image to {output_path_blended}")

            # Convert to RGB for display
            blended_face_result = cv2.cvtColor(blended_face_result, cv2.COLOR_BGR2RGB)
            warped_l_masked = cv2.cvtColor(warped_l_masked, cv2.COLOR_BGR2RGB)
            warped_r_masked = cv2.cvtColor(warped_r_masked, cv2.COLOR_BGR2RGB)
            
            # Create a figure for display
            plt.figure(figsize=(16, 8))
            
            plt.subplot(1, 3, 1)
            plt.imshow(warped_l_masked)
            plt.title('Warped Left Face')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(blended_face_result)
            plt.title('Blended Face (No Lines)')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(warped_r_masked)
            plt.title('Warped Right Face')
            plt.axis('off')

            plt.tight_layout()
            plt.show()


            comparison_results = compare_with_ground_truth(
                    blended_face_result, 
                    ground_truth_path, 
                    blended_mask_result
                )
            # Second row: Comparison with ground truth
            generated_face, ground_truth, diff = comparison_results['images']
            
            plt.subplot(1, 3, 1)
            plt.imshow(ground_truth)
            plt.title('Ground Truth')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(generated_face)
            plt.title('Generated Face')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(diff)
            plt.title(f'Difference Map\nSSIM: {comparison_results["ssim"]:.4f}, PSNR: {comparison_results["psnr"]:.2f}dB')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"!!! An error occurred during file saving or display: {e}")

    else:
        print("Face generation failed, no results to save.")