import cv2
import mediapipe as mp
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay, KDTree
import torch
import os
import sys # Import sys for error exit
import traceback # For detailed error printing

# --- Add local repo path ---
# (Assuming Depth_Anything_V2 is in the same directory or a subdirectory)
LOCAL_DEPTH_ANYTHING_REPO_PATH_RELATIVE = 'Depth_Anything_V2'
LOCAL_DEPTH_ANYTHING_REPO_PATH = os.path.abspath(LOCAL_DEPTH_ANYTHING_REPO_PATH_RELATIVE)
if os.path.isdir(LOCAL_DEPTH_ANYTHING_REPO_PATH):
    sys.path.insert(0, LOCAL_DEPTH_ANYTHING_REPO_PATH)
    print(f"Attempting to add local repo path: {LOCAL_DEPTH_ANYTHING_REPO_PATH}")
    try:
        from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
        print("Successfully imported DepthAnythingV2 class.")
    except ImportError as e:
        print(f"--- !!! ImportError !!! ---"); print(f"Error importing DepthAnythingV2 from: {LOCAL_DEPTH_ANYTHING_REPO_PATH}"); print(f"Specific error: {e}"); print("Ensure path is correct and requirements are installed."); sys.exit("Import Error.")
else:
    print(f"--- !!! Path Not Found Error !!! ---"); print(f"Local repository path not found: {LOCAL_DEPTH_ANYTHING_REPO_PATH}"); sys.exit("Path Error.")


# --- Constants and Configuration ---
NUM_CONTOUR_POINTS = 256
# Percentile for foreground mask (higher value = keep more foreground, as depth is inverted)
DEPTH_THRESHOLD_PERCENTILE = 70 # Adjusted based on inverted depth (higher=closer)
FRONTAL_REF_IMAGE_PATH = 'dataset/1/1/0001.png' # Set path to your frontal reference image
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
DEFAULT_ENCODER = 'vitl'
DEBUG_VISUALIZE = True # Set True to save intermediate debug images

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh

# --- Depth Anything V2 Model Loading ---
def load_depth_anything_model(encoder_type="vitl"):
    # (Function remains the same as your version)
    model_configs = { 'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}, 'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}, 'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}, 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]} }
    if encoder_type not in model_configs: print(f"Warning: Unknown encoder '{encoder_type}'. Using '{DEFAULT_ENCODER}'."); encoder_type = DEFAULT_ENCODER
    weights_path = os.path.join(LOCAL_DEPTH_ANYTHING_REPO_PATH, 'checkpoints', f'depth_anything_v2_{encoder_type}.pth')
    print(f"Loading Depth Anything V2 model ({encoder_type})..."); print(f"  Weights path: {weights_path}")
    try:
        if not os.path.exists(weights_path): raise FileNotFoundError("Weights file not found.")
        model = DepthAnythingV2(**model_configs[encoder_type]); model.load_state_dict(torch.load(weights_path, map_location='cpu')); model = model.to(DEVICE).eval()
        print(f"  Model loaded successfully on {DEVICE}.")
        return model
    except FileNotFoundError as e: print(f"--- Error: {e} ---"); return None
    except Exception as e: print(f"--- Error loading model: {e} ---"); traceback.print_exc(); return None

# --- Depth Estimation (Using model.infer_image) ---
def get_depth_map(model, image_bgr):
    # (Function remains the same as your version, assumes infer_image works)
    if model is None: return None
    print("  Estimating depth map...")
    try:
        depth_map = model.infer_image(image_bgr, input_size=518) # Using input_size 518
        print(f"  Depth map estimated (shape: {depth_map.shape}, range: {depth_map.min():.2f}-{depth_map.max():.2f})")
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min: normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
        else: normalized_depth = np.zeros_like(depth_map)
        inverted_depth = 1.0 - normalized_depth # Higher value = Closer
        print("  Depth map normalized and inverted.")
        return inverted_depth
    except AttributeError: print("--- Error: model object may not have 'infer_image' method."); return None
    except Exception as e: print(f"--- Error during depth estimation: {e} ---"); traceback.print_exc(); return None

# --- Foreground Segmentation (SIMPLIFIED BACK TO PERCENTILE) ---
def get_foreground_mask(inverted_depth_map, percentile=DEPTH_THRESHOLD_PERCENTILE):
    """Creates mask from INVERTED depth map (higher value=closer) using percentile."""
    if inverted_depth_map is None: return None
    print(f"  Generating foreground mask (Percentile: {percentile} -> Threshold on {100-percentile}th value)")
    try:
        # Use percentile on the inverted map (higher values are closer)
        threshold_value = np.percentile(inverted_depth_map, 100 - percentile)
        # Mask where inverted depth is GREATER than threshold (closer objects)
        mask = (inverted_depth_map > threshold_value).astype(np.uint8) * 255
        print(f"    Threshold value: {threshold_value:.3f}")

        # --- Post-processing (Morphological operations) ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)) # Slightly larger kernel?
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3) # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # Remove noise

        # --- Find largest component ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        if num_labels > 1:
             if stats.shape[0] > 1:
                 largest_label_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                 mask = np.where(labels == largest_label_idx, 255, 0).astype(np.uint8)
             else:
                 print("  Warning: No foreground components remain after morphology."); return np.zeros_like(inverted_depth_map, dtype=np.uint8)
        elif num_labels <= 1 and np.sum(mask) == 0:
             print("  Warning: No significant foreground component found in mask."); return np.zeros_like(inverted_depth_map, dtype=np.uint8)

        print("  Foreground mask generated.")
        return mask
    except Exception as e:
        print(f"--- Error creating foreground mask: {e} ---"); traceback.print_exc(); return None

# --- Fallback mask generation using GrabCut (No changes needed) ---
def get_fallback_mask(image, face_landmarks):
    # (Function remains the same as your version)
    if face_landmarks is None or len(face_landmarks) < 3: return None
    print("  Attempting fallback mask generation (GrabCut)...")
    try:
        h, w = image.shape[:2]; face_rect = cv2.boundingRect(face_landmarks.astype(np.int32)); x, y, fw, fh = face_rect
        padding_x = int(fw * 0.4); padding_y1 = int(fh * 0.5); padding_y2 = int(fh * 1.5)
        rect = (max(0, x - padding_x), max(0, y - padding_y1), min(w, fw + 2*padding_x), min(h, fh + padding_y1 + padding_y2))
        gc_mask = np.zeros((h, w), dtype=np.uint8); gc_mask[:,:] = cv2.GC_BGD
        x_r, y_r, w_r, h_r = rect; gc_mask[y_r:y_r+h_r, x_r:x_r+w_r] = cv2.GC_PR_BGD
        hull = cv2.convexHull(face_landmarks.astype(np.int32)); cv2.fillConvexPoly(gc_mask, hull, cv2.GC_FGD) # Mark face as FG
        bgd_model = np.zeros((1, 65), np.float64); fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, gc_mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        final_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)); final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2); final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        print("  Fallback mask generated.")
        return final_mask
    except Exception as e: print(f"--- Error in fallback mask generation: {e} ---"); traceback.print_exc(); return None


# --- Contour Sampling (Fixed max key) ---
def get_contour_keypoints(mask, num_points):
    # (Function remains the same as corrected version)
    if mask is None: return None
    print("  Sampling contour keypoints...")
    try:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours: print("  Warning: No contours found in mask."); return None
        largest_contour = max(contours, key=lambda cnt: cv2.arcLength(cnt, True)) # Fixed
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter < 1e-3: print("  Warning: Contour perimeter too small."); return None
        step = perimeter / num_points; sampled_points = []; distance_accumulated = 0
        sampled_points.append(largest_contour[0][0].astype(np.float32))
        for i in range(len(largest_contour)):
            p1 = largest_contour[i][0]; p2 = largest_contour[(i + 1) % len(largest_contour)][0]
            segment_dist = np.linalg.norm(p1 - p2)
            if segment_dist < 1e-6: continue
            target_num_in_segment = len(sampled_points)
            while len(sampled_points) < num_points and distance_accumulated + segment_dist >= step * target_num_in_segment:
                 required_dist = step * target_num_in_segment
                 if segment_dist < 1e-6: break
                 ratio = (required_dist - distance_accumulated) / segment_dist; ratio = np.clip(ratio, 0, 1)
                 next_sample = p1 + ratio * (p2 - p1); sampled_points.append(next_sample.astype(np.float32)); target_num_in_segment += 1
            if len(sampled_points) >= num_points: break
            distance_accumulated += segment_dist
        while len(sampled_points) < num_points: sampled_points.append(sampled_points[-1])
        final_points = np.array(sampled_points[:num_points], dtype=np.float32)
        print(f"  Sampled {len(final_points)} contour points.")
        return final_points
    except Exception as e: print(f"--- !!! Error sampling contour points !!! ---"); print(f"{e}"); traceback.print_exc(); return None

# --- Face Landmarks (No changes needed) ---
def detect_face_landmarks(image):
    # (Function remains the same)
    if image is None: return None;
    if image.size == 0: return None
    print("  Detecting face landmarks...")
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB));
        if not results.multi_face_landmarks: print("  Warning: No face detected."); return None
        face_landmarks = results.multi_face_landmarks[0]; img_h, img_w, _ = image.shape
        landmarks_px = np.array([(lm.x * img_w, lm.y * img_h) for lm in face_landmarks.landmark], dtype=np.float32)
        print(f"  Detected {len(landmarks_px)} face landmarks.")
        return landmarks_px

# --- Degeneracy Check (no changes needed) ---
def triangle_area(p1, p2, p3):
    # (Function remains the same)
    val = p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]); return val

# --- Warping Function (REVERTED TO ROBUST VERSION) ---
def warp_image_piecewise_affine(src_img, src_points, dst_points, triangles, dst_shape):
    """Warps image using piecewise affine (Robust 'last-on-top' version)."""
    dst_h, dst_w = dst_shape; warped_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8); warped_mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
    num_processed_tris, num_skipped_index, num_skipped_degenerate, num_skipped_affine_error, num_skipped_bbox = 0, 0, 0, 0, 0; DEGENERACY_THRESHOLD = 1e-4
    if triangles is None: print("Warping Error: No triangles."); return warped_img, warped_mask;
    if src_points is None or dst_points is None: print("Warping Error: Missing points."); return warped_img, warped_mask
    if not np.all(np.isfinite(src_points)) or not np.all(np.isfinite(dst_points)): print("!!! WARNING: NaN/Inf in warp points!")
    print(f"  Warping using {len(triangles)} triangles...")
    for i, tri_indices in enumerate(triangles):
        try: tri_indices_list = [int(idx) for idx in tri_indices];
        except (IndexError, ValueError): num_skipped_index += 1; continue;
        if any(idx >= len(src_points) or idx >= len(dst_points) or idx < 0 for idx in tri_indices_list): num_skipped_index += 1; continue
        try: src_tri = src_points[tri_indices_list].astype(np.float32); dst_tri = dst_points[tri_indices_list].astype(np.float32);
        except IndexError: num_skipped_index += 1; continue;
        if not np.all(np.isfinite(src_tri)) or not np.all(np.isfinite(dst_tri)): num_skipped_affine_error += 1; continue
        if src_tri.shape != (3, 2) or dst_tri.shape != (3, 2): num_skipped_affine_error += 1; continue
        src_area = triangle_area(src_tri[0], src_tri[1], src_tri[2]); dst_area = triangle_area(dst_tri[0], dst_tri[1], dst_tri[2])
        if abs(src_area) < DEGENERACY_THRESHOLD or abs(dst_area) < DEGENERACY_THRESHOLD: num_skipped_degenerate += 1; continue
        try: M = cv2.getAffineTransform(src_tri, dst_tri);
        except cv2.error: num_skipped_affine_error += 1; continue;
        if M is None: num_skipped_affine_error += 1; continue
        try: x, y, w, h = cv2.boundingRect(dst_tri.astype(np.int32));
        except cv2.error: num_skipped_bbox += 1; continue;
        x, y, w, h = int(x), int(y), int(w), int(h); x1, y1 = max(0, x), max(0, y); x2, y2 = min(dst_w, x + w), min(dst_h, y + h); w_clip, h_clip = x2 - x1, y2 - y1
        if w_clip <= 0 or h_clip <= 0: num_skipped_bbox += 1; continue
        try:
            bbox_mask = np.zeros((h_clip, w_clip), dtype=np.uint8); dst_tri_rel = dst_tri - np.array([x1, y1], dtype=np.float32)
            cv2.fillConvexPoly(bbox_mask, dst_tri_rel.astype(np.int32), 255, cv2.LINE_AA, 0)
            img_warped_full = cv2.warpAffine(src_img, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            img_warped_patch = img_warped_full[y1:y2, x1:x2]; masked_patch = cv2.bitwise_and(img_warped_patch, img_warped_patch, mask=bbox_mask)
            inv_mask = cv2.bitwise_not(bbox_mask)
            # Overwrite logic (last triangle on top)
            warped_img[y1:y2, x1:x2] = cv2.bitwise_and(warped_img[y1:y2, x1:x2], warped_img[y1:y2, x1:x2], mask=inv_mask)
            warped_img[y1:y2, x1:x2] = cv2.add(warped_img[y1:y2, x1:x2], masked_patch)
            warped_mask[y1:y2, x1:x2] = cv2.bitwise_and(warped_mask[y1:y2, x1:x2], inv_mask)
            warped_mask[y1:y2, x1:x2] = cv2.add(warped_mask[y1:y2, x1:x2], bbox_mask)
        except Exception as e: print(f"!!! ERROR warping/masking triangle {tri_indices}: {e}"); continue
        num_processed_tris += 1
    print(f"  Warping Summary: Processed={num_processed_tris}, Skipped(Index)={num_skipped_index}, Skipped(Degen)={num_skipped_degenerate}, Skipped(Affine)={num_skipped_affine_error}, Skipped(BBox)={num_skipped_bbox}")
    if num_processed_tris == 0: print("  Warning: No triangles were processed during warping.")
    return warped_img, warped_mask


# --- Blending Function (REVERTED TO SIMPLER VERSION) ---
def blend_images_masked_alpha(img1, mask1, img2, mask2):
    """Blends two images using their masks for alpha weighting (Simple version)."""
    print("  Blending warped images...")
    mask1_f = (mask1 / 255.0).astype(np.float32) if mask1.max() > 1 else mask1.astype(np.float32)
    mask2_f = (mask2 / 255.0).astype(np.float32) if mask2.max() > 1 else mask2.astype(np.float32)
    total_alpha = mask1_f + mask2_f
    zero_alpha_mask = total_alpha < 1e-6
    total_alpha[zero_alpha_mask] = 1.0 # Avoid division by zero
    weight1 = mask1_f / total_alpha
    weight2 = mask2_f / total_alpha
    weight1[zero_alpha_mask] = 0.0; weight2[zero_alpha_mask] = 0.0
    img1_f = img1.astype(np.float32); img2_f = img2.astype(np.float32)
    blended_img_f = img1_f * weight1[:, :, np.newaxis] + img2_f * weight2[:, :, np.newaxis]
    combined_mask = (~zero_alpha_mask).astype(np.uint8) * 255
    print("  Blending complete.")
    return blended_img_f.astype(np.uint8), combined_mask

# --- Post Processing (Simplified back) ---
def post_process_foreground(blended_image, mask):
    """Apply basic post-processing."""
    if blended_image is None: return None
    print("  Applying post-processing...")
    processed = blended_image.copy()
    # Apply bilateral filter to smooth while preserving edges
    processed = cv2.bilateralFilter(processed, 9, 75, 75)
    # Optional: Slight blur for remaining artifacts
    # processed = cv2.GaussianBlur(processed, (3, 3), 0.5)
    if mask is not None:
        # Ensure mask is uint8 for bitwise_and
        mask_uint8 = mask.astype(np.uint8)
        mask_3c = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask_uint8
        # Ensure mask_3c is also uint8
        mask_3c = mask_3c.astype(np.uint8)
        processed = cv2.bitwise_and(processed, mask_3c)
    print("  Post-processing complete.")
    return processed

# --- Combined Points (Uses simplified mask function) ---
def get_combined_points(image, depth_model):
    """Detects face and contour points using local Depth Anything V2."""
    print(" Getting combined points...")
    kp_face = detect_face_landmarks(image)
    if kp_face is None: return None, None, None # Return tuple of Nones

    inverted_depth_map = get_depth_map(depth_model, image)
    if inverted_depth_map is None: return None, None, None

    # Use simplified percentile masking first
    mask = get_foreground_mask(inverted_depth_map) # Uses default percentile

    # Try fallback if primary mask fails or is too small
    if mask is None or np.sum(mask) < 1000: # Check if mask is too small
        print("  Warning: Primary mask generation failed or mask too small. Trying fallback...")
        mask = get_fallback_mask(image, kp_face)

    if mask is None or np.sum(mask) < 1000: # Check fallback result
        print("  Warning: Fallback mask generation also failed or mask too small.")
        return None, None, None # Cannot proceed without a mask

    kp_contour = get_contour_keypoints(mask, NUM_CONTOUR_POINTS)
    if kp_contour is None: return None, None, None

    print(" Combined points acquired.")
    return kp_face, kp_contour, mask

# --- Correspondence (No changes needed) ---
def establish_correspondence(kp_l_face, kp_l_contour, kp_r_face, kp_r_contour, kp_f_face, kp_f_contour):
    # (Function remains the same as corrected version)
    if kp_l_face is None or kp_r_face is None or kp_f_face is None or kp_l_contour is None or kp_r_contour is None or kp_f_contour is None: print("Error: Missing points for correspondence."); return None, None
    print("  Establishing correspondence...")
    try:
        T_L_to_F_mat, inliers_l = cv2.estimateAffinePartial2D(kp_l_face, kp_f_face, method=cv2.RANSAC); T_R_to_F_mat, inliers_r = cv2.estimateAffinePartial2D(kp_r_face, kp_f_face, method=cv2.RANSAC);
        if T_L_to_F_mat is None or T_R_to_F_mat is None: print("  Error: Failed alignment transform."); return None, None
        print(f"    Face alignment inliers: L={np.sum(inliers_l) if inliers_l is not None else 'N/A'}, R={np.sum(inliers_r) if inliers_r is not None else 'N/A'}")
        kp_l_contour_hom = np.hstack((kp_l_contour, np.ones((len(kp_l_contour), 1)))); kp_l_contour_frontal_space = (T_L_to_F_mat @ kp_l_contour_hom.T).T
        kp_r_contour_hom = np.hstack((kp_r_contour, np.ones((len(kp_r_contour), 1)))); kp_r_contour_frontal_space = (T_R_to_F_mat @ kp_r_contour_hom.T).T
        tree_frontal = KDTree(kp_f_contour); _, l_indices_in_frontal = tree_frontal.query(kp_l_contour_frontal_space); _, r_indices_in_frontal = tree_frontal.query(kp_r_contour_frontal_space)
        kp_l_contour_ordered = np.zeros_like(kp_f_contour); kp_r_contour_ordered = np.zeros_like(kp_f_contour); counts_l = np.zeros(len(kp_f_contour), dtype=int); counts_r = np.zeros(len(kp_f_contour), dtype=int)
        for i, frontal_idx in enumerate(l_indices_in_frontal):
            if 0 <= frontal_idx < len(kp_l_contour_ordered): kp_l_contour_ordered[frontal_idx] += kp_l_contour[i]; counts_l[frontal_idx] += 1
        for i, frontal_idx in enumerate(r_indices_in_frontal):
             if 0 <= frontal_idx < len(kp_r_contour_ordered): kp_r_contour_ordered[frontal_idx] += kp_r_contour[i]; counts_r[frontal_idx] += 1
        unmatched_l_count=0; unmatched_r_count=0
        for k in range(len(kp_f_contour)):
            if counts_l[k] > 0: kp_l_contour_ordered[k] /= counts_l[k]
            else: unmatched_l_count+=1; kp_l_contour_ordered[k] = kp_f_contour[k]
            if counts_r[k] > 0: kp_r_contour_ordered[k] /= counts_r[k]
            else: unmatched_r_count+=1; kp_r_contour_ordered[k] = kp_f_contour[k]
        if unmatched_l_count > 0: print(f"    Warning: {unmatched_l_count}/{len(kp_f_contour)} L contour points unmatched.")
        if unmatched_r_count > 0: print(f"    Warning: {unmatched_r_count}/{len(kp_f_contour)} R contour points unmatched.")
        final_kp_l = np.vstack((kp_l_face, kp_l_contour_ordered)); final_kp_r = np.vstack((kp_r_face, kp_r_contour_ordered))
        print("  Correspondence established.")
        return final_kp_l, final_kp_r
    except Exception as e: print(f"--- !!! Error during correspondence !!! ---"); print(f"{e}"); traceback.print_exc(); return None, None


# --- Main Function (Simplified Control Flow) ---
def generate_intermediate_foreground(img_l_path, img_r_path, img_f_path, depth_model):
    """Generates intermediate view using local Depth Anything V2."""
    print("-" * 30); print(f"Starting Processing:"); print(f" L='{os.path.basename(img_l_path)}'"); print(f" R='{os.path.basename(img_r_path)}'"); print(f" F='{os.path.basename(img_f_path)}'"); print("(Using Local Depth Anything V2)"); print("-" * 30)

    print("Loading images...")
    img_l = cv2.imread(img_l_path); img_r = cv2.imread(img_r_path); img_f = cv2.imread(img_f_path)
    if img_l is None or img_r is None or img_f is None: print("Error loading images."); return None
    print(" Images loaded.")

    # Create debug directory if requested
    if DEBUG_VISUALIZE:
        debug_dir = "debug_output"; os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug images will be saved to: {debug_dir}")

    # Process Views
    view_data = {}
    processing_ok = True
    for name, img in [("frontal", img_f), ("left", img_l), ("right", img_r)]:
        print(f"\n--- Processing {name.capitalize()} View ---")
        kp_face, kp_contour, mask = get_combined_points(img, depth_model)
        if kp_face is None or kp_contour is None or mask is None:
            print(f"!!! Critical Error: Failed to process {name.capitalize()} View. Cannot proceed."); processing_ok = False; break
        view_data[name] = {'kp_face': kp_face, 'kp_contour': kp_contour, 'mask': mask}
        print(f"--- {name.capitalize()} View Processing Complete ---")
        # Save debug mask
        if DEBUG_VISUALIZE and mask is not None:
             cv2.imwrite(os.path.join(debug_dir, f"{name}_mask_final.png"), mask)

    if not processing_ok: return None

    # Establish Correspondence
    print("\n--- Establishing Correspondence ---")
    kp_l, kp_r = establish_correspondence(
        view_data['left']['kp_face'], view_data['left']['kp_contour'],
        view_data['right']['kp_face'], view_data['right']['kp_contour'],
        view_data['frontal']['kp_face'], view_data['frontal']['kp_contour']
    )
    if kp_l is None or kp_r is None: print("!!! Critical Error: Failed correspondence."); return None
    if len(kp_l) != len(kp_r): print("!!! Error: Point count mismatch."); return None
    print("--- Correspondence Complete ---")

    # Intermediate Geometry
    print("\n--- Calculating Intermediate Geometry ---")
    kp_intermediate = 0.5 * kp_l + 0.5 * kp_r
    if not np.all(np.isfinite(kp_intermediate)): print("!!! ERROR: NaN/Inf in intermediate points!"); return None
    print("  Intermediate keypoints calculated.")
    triangles = None
    try: delaunay = Delaunay(kp_intermediate); triangles = delaunay.simplices; print(f"  Calculated {len(triangles)} Delaunay triangles.")
    except Exception as e: print(f"!!! ERROR during Delaunay: {e}"); return None
    # Debug Triangulation
    if DEBUG_VISUALIZE and triangles is not None:
         h_tri, w_tri = img_l.shape[:2]; tri_img = np.zeros((h_tri, w_tri, 3), dtype=np.uint8)
         for tri in triangles: pts = kp_intermediate[tri].astype(np.int32); cv2.polylines(tri_img, [pts], True, (0, 255, 0), 1)
         for pt in kp_intermediate: cv2.circle(tri_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
         cv2.imwrite(os.path.join(debug_dir, "triangulation.png"), tri_img)
    print("--- Intermediate Geometry Complete ---")

    # Warping
    print("\n--- Warping Images ---")
    h, w = img_l.shape[:2]; dst_shape = (h, w)
    print(" Warping left image...")
    warped_l_fg, warped_mask_l = warp_image_piecewise_affine(img_l, kp_l, kp_intermediate, triangles, dst_shape)
    print(" Warping right image...")
    warped_r_fg, warped_mask_r = warp_image_piecewise_affine(img_r, kp_r, kp_intermediate, triangles, dst_shape)
    # Debug Warped
    if DEBUG_VISUALIZE:
         cv2.imwrite(os.path.join(debug_dir, "warped_left.png"), warped_l_fg)
         cv2.imwrite(os.path.join(debug_dir, "warped_left_mask.png"), warped_mask_l)
         cv2.imwrite(os.path.join(debug_dir, "warped_right.png"), warped_r_fg)
         cv2.imwrite(os.path.join(debug_dir, "warped_right_mask.png"), warped_mask_r)
    print("--- Warping Complete ---")

    # Blending & Post-processing
    print("\n--- Blending & Post-processing ---")
    blended_fg, blended_mask = blend_images_masked_alpha(warped_l_fg, warped_mask_l, warped_r_fg, warped_mask_r)
    # Debug Blended
    if DEBUG_VISUALIZE:
         cv2.imwrite(os.path.join(debug_dir, "blended_raw.png"), blended_fg)
         cv2.imwrite(os.path.join(debug_dir, "blended_mask.png"), blended_mask)
    final_result = post_process_foreground(blended_fg, blended_mask)
     # Debug Final
    if DEBUG_VISUALIZE:
         cv2.imwrite(os.path.join(debug_dir, "final_processed.png"), final_result)
    print("--- Blending & Post-processing Complete ---")

    print("-" * 30); print("Processing Finished Successfully."); print("-" * 30)
    return final_result, blended_mask, warped_l_fg, warped_mask_l, warped_r_fg, warped_mask_r, view_data['left']['mask'], view_data['right']['mask'], view_data['frontal']['mask']

# --- Example Usage ---
if __name__ == "__main__":
    print("Script starting...")
    # ==================================================
    left_image_path = 'dataset/1/1/0003.png'
    right_image_path = 'dataset/1/1/0002.png'
    frontal_image_path = FRONTAL_REF_IMAGE_PATH # Use constant
    output_base = "intermediate_foreground_local_DA_output"
    # ==================================================

    # --- Load Depth Anything V2 Model ---
    chosen_encoder = 'vits' # vits, vitb, vitl, vitg
    print(f"Loading depth model (Encoder: {chosen_encoder})...")
    depth_model = load_depth_anything_model(chosen_encoder)

    if depth_model is None:
        print("\nFailed to load Depth Anything V2 model. Exiting.")
    else:
        print("\nStarting main processing pipeline...")
        # --- Run the main generation function ---
        # We need the intermediate points from it, so we call it first.
        # The function itself returns the final image, but we need earlier data for vis.

        print("-" * 30); print(f"Starting Processing:"); print(f" L='{os.path.basename(left_image_path)}'"); print(f" R='{os.path.basename(right_image_path)}'"); print(f" F='{os.path.basename(frontal_image_path)}'"); print("(Using Local Depth Anything V2)"); print("-" * 30)

        print("Loading images...")
        img_l = cv2.imread(left_image_path); img_r = cv2.imread(right_image_path); img_f = cv2.imread(frontal_image_path)
        if img_l is None or img_r is None or img_f is None: print("Error loading images."); sys.exit()
        print(" Images loaded.")

        if DEBUG_VISUALIZE: debug_dir = "debug_output"; os.makedirs(debug_dir, exist_ok=True)

        # --- Process Views ---
        view_data = {}; processing_ok = True
        for name, img in [("frontal", img_f), ("left", img_l), ("right", img_r)]:
            print(f"\n--- Processing {name.capitalize()} View ---")
            kp_face, kp_contour, mask = get_combined_points(img, depth_model)
            if kp_face is None or kp_contour is None or mask is None: print(f"!!! Critical Error: Failed {name.capitalize()} View."); processing_ok = False; break
            view_data[name] = {'kp_face': kp_face, 'kp_contour': kp_contour, 'mask': mask}
            print(f"--- {name.capitalize()} View Complete ---")
            if DEBUG_VISUALIZE and mask is not None: cv2.imwrite(os.path.join(debug_dir, f"{name}_mask_final.png"), mask)
        if not processing_ok: sys.exit("Processing stopped.")

        # --- Establish Correspondence ---
        print("\n--- Establishing Correspondence ---")
        kp_l, kp_r = establish_correspondence(view_data['left']['kp_face'], view_data['left']['kp_contour'], view_data['right']['kp_face'], view_data['right']['kp_contour'], view_data['frontal']['kp_face'], view_data['frontal']['kp_contour'])
        if kp_l is None or kp_r is None: print("!!! Critical Error: Failed correspondence."); sys.exit()
        if len(kp_l) != len(kp_r): print("!!! Error: Point count mismatch."); sys.exit()
        print("--- Correspondence Complete ---")

        # --- Calculate Intermediate Geometry ---
        print("\n--- Calculating Intermediate Geometry ---")
        kp_intermediate = 0.5 * kp_l + 0.5 * kp_r
        if not np.all(np.isfinite(kp_intermediate)): print("!!! ERROR: NaN/Inf in intermediate points!"); sys.exit()
        print("  Intermediate keypoints calculated.")
        triangles = None
        try: delaunay = Delaunay(kp_intermediate); triangles = delaunay.simplices; print(f"  Calculated {len(triangles)} Delaunay triangles.")
        except Exception as e: print(f"!!! ERROR during Delaunay: {e}"); sys.exit()
        print("--- Intermediate Geometry Complete ---")

        # --- !!! VISUALIZATION OF INTERMEDIATE POINTS !!! ---
        print("\n--- Visualizing Intermediate Points ---")
        try:
            # Choose a background image (e.g., left input)
            vis_background = img_l.copy()
            num_face_points = len(view_data['frontal']['kp_face']) # Get count from reference

            # Draw Contour Points (e.g., green) - indices >= num_face_points
            for i in range(num_face_points, len(kp_intermediate)):
                pt = kp_intermediate[i]
                cv2.circle(vis_background, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1) # Green circle

            # Draw Face Points (e.g., red) - indices < num_face_points
            for i in range(num_face_points):
                pt = kp_intermediate[i]
                cv2.circle(vis_background, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1) # Smaller red circle

            # Optionally draw the triangulation on top
            if triangles is not None:
                 print("  Drawing triangulation overlay...")
                 for tri in triangles:
                     pts = kp_intermediate[tri].astype(np.int32)
                     cv2.polylines(vis_background, [pts], True, (255, 255, 0), 1, cv2.LINE_AA) # Cyan lines

            # Display using Matplotlib
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(vis_background, cv2.COLOR_BGR2RGB))
            plt.title(f'Intermediate Keypoints (Face={num_face_points}, Contour={NUM_CONTOUR_POINTS}) + Triangulation')
            plt.axis('off')
            if DEBUG_VISUALIZE:
                vis_path = os.path.join(debug_dir, "intermediate_points_visualization.png")
                cv2.imwrite(vis_path, vis_background)
                print(f"  Saved points visualization to: {vis_path}")
            plt.show()

        except Exception as e:
            print(f"--- Error during points visualization: {e} ---")
            traceback.print_exc()
        # --- END VISUALIZATION ---


        # --- Continue with the rest of the process (Warping, Blending, etc.) ---
        # (You might want to comment out the final display if you only want point viz now)
        print("\n--- Warping Images ---")
        h, w = img_l.shape[:2]; dst_shape = (h, w)
        warped_l_fg, warped_mask_l = warp_image_piecewise_affine(img_l, kp_l, kp_intermediate, triangles, dst_shape)
        warped_r_fg, warped_mask_r = warp_image_piecewise_affine(img_r, kp_r, kp_intermediate, triangles, dst_shape)
        if DEBUG_VISUALIZE: cv2.imwrite(os.path.join(debug_dir, "warped_left.png"), warped_l_fg); cv2.imwrite(os.path.join(debug_dir, "warped_right.png"), warped_r_fg)
        print("--- Warping Complete ---")

        print("\n--- Blending & Post-processing ---")
        blended_fg, blended_mask = blend_images_masked_alpha(warped_l_fg, warped_mask_l, warped_r_fg, warped_mask_r)
        if DEBUG_VISUALIZE: cv2.imwrite(os.path.join(debug_dir, "blended_raw.png"), blended_fg); cv2.imwrite(os.path.join(debug_dir, "blended_mask.png"), blended_mask)
        final_result = post_process_foreground(blended_fg, blended_mask)
        if DEBUG_VISUALIZE: cv2.imwrite(os.path.join(debug_dir, "final_processed.png"), final_result)
        print("--- Blending & Post-processing Complete ---")

        plt.imshow(final_result)
        plt.show()

        plt.imshow(blended_fg)
        plt.show()

        print("-" * 30); print("Processing Finished Successfully."); print("-" * 30)

        # --- Display Final Results (Optional) ---
        # try:
        #     print("Displaying final results...")
        #     plt.figure(figsize=(16, 9))
        #     # ... (your original final display code) ...
        #     plt.show()
        # except Exception as e: print(f"!!! An error occurred during display: {e}")


    print("\nScript finished.")