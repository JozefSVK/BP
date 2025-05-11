import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import time

# --- Konfigurácia ---
IMAGE_PATH_A = 'dataset/1/1/0003.png'  # ! NASTAV CESTU K LAVEMU OBRAZKU
IMAGE_PATH_B = 'dataset/1/1/0002.png'  # ! NASTAV CESTU K PRAVEMU OBRAZKU
SHOW_PLOTS = True
USE_DEPTH = False  # Možnosť použiť segmentáciu pomocou hĺbky (ak je k dispozícii kód)

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh

# --- Definícia Landmarkov ---
# Selektívne vybrané body pre lepšie morfovanie bez artefaktov v strede tváre

# Kľúčové body tváre (bez problematických kontúr)
# Oči, nos, ústa, obočie - hlavné črty bez okrajov
EYES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

EYEBROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 285, 295, 282, 283, 276]

NOSE = [1, 2, 98, 327, 358, 19, 20, 44, 45, 275, 4, 5, 195, 197, 6, 168, 
        48, 219, 237, 44, 2, 97, 141, 99, 97, 2, 326, 328, 330]

MOUTH = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 
        0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Okraje tváre bez stredových bodov (iba boky, brada a horná čast čela)
# Vyhýbame sa bodom v strede čela a na nose, ktoré môžu spôsobovať artefakty
PARTIAL_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Vonkajšie kontúry hlavy (uši, čeľusť, temeno hlavy)
# Tu by sme mohli dodať body z hĺbkovej mapy, ak je k dispozícii
BACK_HEAD = []  # Bude nahradené bodmi z hĺbkovej segmentácie

# Spojené body tváre bez problémových landmarkov
FACE_POINTS = EYES + EYEBROWS + NOSE + MOUTH + PARTIAL_FACE_OVAL
# Odstránenie duplikátov
FACE_POINTS = list(set(FACE_POINTS))

# --- Helper Functions ---

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

def extract_selected_landmarks(all_landmarks, indices):
    """Extract only selected landmarks by indices."""
    return np.array([all_landmarks[i] for i in indices if i < len(all_landmarks)], dtype=np.float32)

def segment_head(image, face_landmarks):
    """
    Create a rough head segmentation mask using face landmarks.
    In a real implementation, this would use depth estimation for better results.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Extract face oval points for the basic mask
    face_oval_points = []
    for idx in PARTIAL_FACE_OVAL:
        if idx < len(face_landmarks):
            face_oval_points.append(face_landmarks[idx])
    
    if face_oval_points:
        # Convert to numpy array for OpenCV
        face_oval_points = np.array(face_oval_points, dtype=np.int32)
        
        # First create the basic face mask
        cv2.fillConvexPoly(mask, face_oval_points, 255)
        
        # Expand the mask with a dilate operation to include more of the head
        # Toto je veľmi jednoduché - reálna implementácia by použila hĺbkový odhad
        kernel = np.ones((30, 30), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        
        # Smooth the mask edges
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
    
    return mask

def get_head_contour_points(mask, num_points=20):
    """Extract points along the contour of the head segmentation mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.array([])
    
    # Find the largest contour (should be the head)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Sample evenly spaced points along the contour
    length = cv2.arcLength(largest_contour, closed=True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon=length/num_points, closed=True)
    
    # Convert to the right format
    contour_points = approx_contour.reshape(-1, 2).astype(np.float32)
    
    return contour_points

def triangle_area(p1, p2, p3):
    """Calculate the area of a triangle given its vertices."""
    return 0.5 * abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])))

def warp_image_piecewise_affine(src_img, src_points, dst_points, triangles, dst_shape):
    """
    Warps an image using piecewise affine transformation based on triangle meshes.
    Enhanced to handle better triangle edges and overlaps.
    """
    dst_h, dst_w = dst_shape
    warped_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    warped_mask = np.zeros((dst_h, dst_w), dtype=np.float32)  # Using float for smoother blending
    
    # Collect all warped triangles for better blending
    warped_triangles = []

    for i, tri_indices in enumerate(triangles):
        try:
            # Ensure indices are integers
            tri_indices = [int(idx) for idx in tri_indices]
            
            # Check if any index is out of bounds
            if any(idx >= len(src_points) or idx >= len(dst_points) or idx < 0 for idx in tri_indices):
                continue
                
            src_tri = np.array([src_points[idx] for idx in tri_indices], dtype=np.float32)
            dst_tri = np.array([dst_points[idx] for idx in tri_indices], dtype=np.float32)
            
            # Check for degenerate triangles
            if src_tri.shape != (3, 2) or dst_tri.shape != (3, 2):
                continue
                
            # Check for very small triangles
            area = triangle_area(dst_tri[0], dst_tri[1], dst_tri[2])
            if area < 1.0:  # Skip very small triangles
                continue
                
            # Calculate bounding rectangle for destination triangle
            r = cv2.boundingRect(dst_tri.astype(np.int32))
            (x, y, w_rect, h_rect) = r
            
            # Make sure we're not going out of bounds
            if x < 0 or y < 0 or x + w_rect > dst_w or y + h_rect > dst_h:
                # Adjust bounding box to be within image
                x_start = max(0, x)
                y_start = max(0, y)
                x_end = min(dst_w, x + w_rect)
                y_end = min(dst_h, y + h_rect)
                w_rect = x_end - x_start
                h_rect = y_end - y_start
                x = x_start
                y = y_start
                
                if w_rect <= 0 or h_rect <= 0:
                    continue  # Skip this triangle if it's outside the image
            
            # Offset destination triangle to be relative to the bounding box
            dst_tri_offset = [(dst_tri[i][0] - x, dst_tri[i][1] - y) for i in range(3)]
            dst_tri_offset = np.array(dst_tri_offset, dtype=np.float32)
            
            # Compute affine transform
            M = cv2.getAffineTransform(src_tri, dst_tri_offset)
            
            # Warp the triangle region from source to rectangle
            warped_triangle = cv2.warpAffine(src_img, M, (w_rect, h_rect), 
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REPLICATE)
            
            # Create a mask for the triangle with anti-aliasing
            mask = np.zeros((h_rect, w_rect), dtype=np.float32)
            cv2.fillConvexPoly(mask, dst_tri_offset.astype(np.int32), 1.0, cv2.LINE_AA)
            
            # Apply a slight Gaussian blur for smoother edges
            mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
            
            # Store the warped triangle for later blending
            warped_triangles.append((warped_triangle, mask, (x, y, w_rect, h_rect)))
            
        except Exception as e:
            print(f"Error processing triangle {i}: {e}")
            continue
    
    # Now blend all warped triangles with proper weighting
    for warped_triangle, mask, (x, y, w_rect, h_rect) in warped_triangles:
        try:
            # Get the current region in the output image
            roi = warped_img[y:y+h_rect, x:x+w_rect].astype(np.float32)
            roi_mask = warped_mask[y:y+h_rect, x:x+w_rect]
            
            # Calculate weights for blending
            combined_mask = roi_mask + mask
            # Avoid division by zero
            combined_mask[combined_mask == 0] = 1.0
            
            # Calculate normalized weights
            w1 = roi_mask / combined_mask
            w2 = mask / combined_mask
            
            # Blend the colors
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * w1 + warped_triangle[:, :, c] * w2
            
            # Update the output image and mask
            warped_img[y:y+h_rect, x:x+w_rect] = roi.astype(np.uint8)
            warped_mask[y:y+h_rect, x:x+w_rect] = np.maximum(roi_mask, mask)
            
        except Exception as e:
            print(f"Error blending triangle: {e}")
            continue
    
    # Convert the mask back to uint8
    return warped_img, (warped_mask * 255).astype(np.uint8)

def create_blended_image(warped_A, mask_A, warped_B, mask_B, alpha=0.5):
    """Create a blended image with proper alpha weighting and mask handling."""
    # Convert masks to float [0,1]
    mask_A_f = mask_A.astype(float) / 255.0
    mask_B_f = mask_B.astype(float) / 255.0
    
    # Apply Gaussian blur to masks for smoother transitions
    mask_A_f = cv2.GaussianBlur(mask_A_f, (5, 5), 1.0)
    mask_B_f = cv2.GaussianBlur(mask_B_f, (5, 5), 1.0)
    
    # Calculate combined weight
    combined_mask = mask_A_f + mask_B_f
    # Avoid division by zero
    combined_mask[combined_mask == 0] = 1.0
    
    # Calculate normalized weights with alpha
    w1 = mask_A_f / combined_mask * alpha
    w2 = mask_B_f / combined_mask * (1.0 - alpha)
    
    # Convert to 3-channel for image blending
    w1_3c = np.stack([w1, w1, w1], axis=2)
    w2_3c = np.stack([w2, w2, w2], axis=2)
    
    # Blend images using the weights
    blended = (warped_A.astype(float) * w1_3c + warped_B.astype(float) * w2_3c).astype(np.uint8)
    
    return blended

def post_process_image(image, mask=None):
    """Apply post-processing to improve the final result."""
    # Create a copy to work on
    result = image.copy()
    
    # Apply bilateral filter to smooth while preserving edges
    result = cv2.bilateralFilter(result, 7, 35, 35)
    
    # Apply a very light Gaussian blur to further reduce noise
    result = cv2.GaussianBlur(result, (3, 3), 0.5)
    
    # Apply the mask if provided
    if mask is not None:
        if len(mask.shape) == 2:  # Convert single-channel mask to 3-channel
            mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_3c = mask
        # Ensure correct data type
        mask_3c = mask_3c.astype(np.uint8)
        result = cv2.bitwise_and(result, mask_3c)
    
    return result

def main():
    # Load images
    print("Loading images...")
    img_A_bgr = cv2.imread(IMAGE_PATH_A)
    img_B_bgr = cv2.imread(IMAGE_PATH_B)
    
    if img_A_bgr is None or img_B_bgr is None:
        print("Error: Could not load images.")
        return
    
    # Get original dimensions
    h, w = img_A_bgr.shape[:2]
    
    # Convert to RGB for processing and visualization
    img_A_rgb = cv2.cvtColor(img_A_bgr, cv2.COLOR_BGR2RGB)
    img_B_rgb = cv2.cvtColor(img_B_bgr, cv2.COLOR_BGR2RGB)
    
    # Detect all face landmarks
    print("Detecting landmarks...")
    all_landmarks_A = detect_face_landmarks(img_A_bgr)
    all_landmarks_B = detect_face_landmarks(img_B_bgr)
    
    if all_landmarks_A is None or all_landmarks_B is None:
        print("Error: Could not detect faces in both images.")
        return
    
    # Extract only the selected face points (avoiding problematic contours)
    print("Extracting selected landmarks...")
    face_points_A = extract_selected_landmarks(all_landmarks_A, FACE_POINTS)
    face_points_B = extract_selected_landmarks(all_landmarks_B, FACE_POINTS)
    
    print(f"Using {len(face_points_A)} face landmark points.")
    
    # Create head segmentation masks
    print("Creating head segmentation...")
    head_mask_A = segment_head(img_A_rgb, all_landmarks_A)
    head_mask_B = segment_head(img_B_rgb, all_landmarks_B)
    
    # Extract head contour points
    print("Extracting head contour points...")
    head_contour_A = get_head_contour_points(head_mask_A)
    head_contour_B = get_head_contour_points(head_mask_B)
    
    print(f"Found {len(head_contour_A)} points for head contour A")
    print(f"Found {len(head_contour_B)} points for head contour B")
    
    # Combine face points with head contour points
    all_points_A = np.vstack([face_points_A, head_contour_A]) if len(head_contour_A) > 0 else face_points_A
    all_points_B = np.vstack([face_points_B, head_contour_B]) if len(head_contour_B) > 0 else face_points_B
    
    # Calculate midway points
    mid_points = (all_points_A + all_points_B) / 2.0
    
    # Perform Delaunay triangulation on the mid-points
    print("Performing Delaunay triangulation...")
    tri = Delaunay(mid_points)
    
    # Warp images using the enhanced warping function
    print("Warping image A to midway points...")
    warped_A, mask_A = warp_image_piecewise_affine(
        img_A_rgb, all_points_A, mid_points, tri.simplices, (h, w))
    
    print("Warping image B to midway points...")
    warped_B, mask_B = warp_image_piecewise_affine(
        img_B_rgb, all_points_B, mid_points, tri.simplices, (h, w))
    
    # Create the combined head mask
    combined_mask = cv2.bitwise_or(mask_A, mask_B)
    combined_mask = cv2.bitwise_or(combined_mask, cv2.bitwise_or(head_mask_A, head_mask_B))
    
    # Smooth the combined mask
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 1.0)
    
    # Create a properly blended image
    print("Blending warped images...")
    blended = create_blended_image(warped_A, mask_A, warped_B, mask_B, alpha=0.5)
    
    # Apply post-processing
    print("Applying post-processing...")
    final_result = post_process_image(blended, combined_mask)
    
    # For comparison, also create a simple blended version
    simple_blend = cv2.addWeighted(
        warped_A, 0.5, 
        warped_B, 0.5, 
        0
    )
    
    # Visualize results
    if SHOW_PLOTS:
        print("Displaying results...")
        plt.figure(figsize=(15, 10))
        
        # Show original images with landmarks
        plt.subplot(2, 3, 1)
        plt.imshow(img_A_rgb)
        plt.scatter(all_points_A[:, 0], all_points_A[:, 1], color='red', s=1)
        plt.title('Image A with Points')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(img_B_rgb)
        plt.scatter(all_points_B[:, 0], all_points_B[:, 1], color='red', s=1)
        plt.title('Image B with Points')
        plt.axis('off')
        
        # Show warped images and final result
        plt.subplot(2, 3, 4)
        plt.imshow(warped_A)
        plt.title('Warped A')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(final_result)
        plt.title('Final Morphed Result')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(warped_B)
        plt.title('Warped B')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Compare simple vs. advanced blending
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(simple_blend)
        plt.title('Simple Blending')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(blended)
        plt.title('Advanced Blending')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(final_result)
        plt.title('Final Result (Post-processed)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Show masks
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(mask_A, cmap='gray')
        plt.title('Warp Mask A')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(combined_mask, cmap='gray')
        plt.title('Combined Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(mask_B, cmap='gray')
        plt.title('Warp Mask B')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("Processing complete!")
    return final_result, warped_A, warped_B

if __name__ == "__main__":
    main()