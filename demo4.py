import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# --- Konfigurácia ---
IMAGE_PATH_A = 'dataset/1/1/0003.png'  # ! NASTAV CESTU K LAVEMU OBRAZKU
IMAGE_PATH_B = 'dataset/1/1/0002.png'  # ! NASTAV CESTU K PRAVEMU OBRAZKU
SHOW_PLOTS = True

# --- Inicializácia MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

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

def warp_image_piecewise_affine(src_img, src_points, dst_points, triangles, dst_shape):
    """
    Warps an image using piecewise affine transformation based on triangle meshes.
    """
    dst_h, dst_w = dst_shape
    warped_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    warped_mask = np.zeros((dst_h, dst_w), dtype=np.uint8)

    for i, tri_indices in enumerate(triangles):
        # Get vertices
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
                
            # Calculate bounding rectangle for destination triangle
            r = cv2.boundingRect(dst_tri.astype(np.int32))
            (x, y, w_rect, h_rect) = r
            
            # Offset destination triangle
            dst_tri_offset = [(dst_tri[i][0] - x, dst_tri[i][1] - y) for i in range(3)]
            dst_tri_offset = np.array(dst_tri_offset, dtype=np.float32)
            
            # Compute affine transform
            M = cv2.getAffineTransform(src_tri, dst_tri_offset)
            
            # Warp the triangle region from source to rectangle
            warped_triangle = cv2.warpAffine(src_img, M, (w_rect, h_rect), borderMode=cv2.BORDER_REPLICATE)
            
            # Create a mask for the triangle
            mask = np.zeros((h_rect, w_rect), dtype=np.uint8)
            cv2.fillConvexPoly(mask, dst_tri_offset.astype(np.int32), 255)
            
            # Apply the mask
            warped_triangle_masked = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask)
            
            # Copy the warped triangle to the output image
            # First create a ROI in the output image
            roi = warped_img[y:y+h_rect, x:x+w_rect]
            
            # Combine with existing pixels using the inverse mask
            inv_mask = cv2.bitwise_not(mask)
            roi_bg = cv2.bitwise_and(roi, roi, mask=inv_mask)
            
            # Add the warped triangle to the ROI
            dst_roi = cv2.add(roi_bg, warped_triangle_masked)
            warped_img[y:y+h_rect, x:x+w_rect] = dst_roi
            
            # Update the mask to indicate warped regions
            warped_mask[y:y+h_rect, x:x+w_rect] = cv2.bitwise_or(
                warped_mask[y:y+h_rect, x:x+w_rect], mask)
                
        except Exception as e:
            print(f"Error processing triangle {i}: {e}")
            continue
            
    return warped_img, warped_mask

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
    
    # Detect face landmarks
    print("Detecting landmarks...")
    face_points_A = detect_face_landmarks(img_A_bgr)
    face_points_B = detect_face_landmarks(img_B_bgr)
    
    if face_points_A is None or face_points_B is None:
        print("Error: Could not detect faces in both images.")
        return
    
    # Use a subset of landmarks to focus on the facial features
    # Using every 5th landmark to simplify
    indices = list(range(0, len(face_points_A), 5))
    face_points_A_subset = face_points_A[indices]
    face_points_B_subset = face_points_B[indices]
    
    print(f"Using {len(face_points_A_subset)} landmark points for each face.")
    
    # Calculate midway points
    mid_points = (face_points_A_subset + face_points_B_subset) / 2.0
    
    # Perform Delaunay triangulation on the mid-points
    print("Performing Delaunay triangulation...")
    tri = Delaunay(mid_points)
    
    # Convert images to RGB for visualization
    img_A_rgb = cv2.cvtColor(img_A_bgr, cv2.COLOR_BGR2RGB)
    img_B_rgb = cv2.cvtColor(img_B_bgr, cv2.COLOR_BGR2RGB)
    
    # Warp images
    print("Warping image A to midway points...")
    warped_A, mask_A = warp_image_piecewise_affine(
        img_A_rgb, face_points_A_subset, mid_points, tri.simplices, (h, w))
    
    print("Warping image B to midway points...")
    warped_B, mask_B = warp_image_piecewise_affine(
        img_B_rgb, face_points_B_subset, mid_points, tri.simplices, (h, w))
    
    # Create face masks (simplified for debugging)
    face_mask_A = np.ones((h, w), dtype=np.uint8) * 255 
    face_mask_B = np.ones((h, w), dtype=np.uint8) * 255
    
    # Apply face masks to warped images
    warped_A_masked = cv2.bitwise_and(warped_A, warped_A, mask=mask_A)
    warped_B_masked = cv2.bitwise_and(warped_B, warped_B, mask=mask_B)
    
    # Simple alpha blending for face morphing
    alpha = 0.5
    morphed = cv2.addWeighted(warped_A_masked, alpha, warped_B_masked, 1-alpha, 0)
    
    # Visualize results
    if SHOW_PLOTS:
        print("Displaying results...")
        plt.figure(figsize=(15, 10))
        
        # Show original images with landmarks
        plt.subplot(2, 3, 1)
        plt.imshow(img_A_rgb)
        plt.scatter(face_points_A_subset[:, 0], face_points_A_subset[:, 1], color='red', s=1)
        plt.title('Image A with Landmarks')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(img_B_rgb)
        plt.scatter(face_points_B_subset[:, 0], face_points_B_subset[:, 1], color='red', s=1)
        plt.title('Image B with Landmarks')
        plt.axis('off')
        
        # Show warped images and morphed result
        plt.subplot(2, 3, 4)
        plt.imshow(warped_A_masked)
        plt.title('Warped A')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(morphed)
        plt.title('Morphed Result')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        plt.imshow(warped_B_masked)
        plt.title('Warped B')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Debug: Show masks
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(mask_A, cmap='gray')
        plt.title('Mask A')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask_B, cmap='gray')
        plt.title('Mask B')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return morphed, warped_A_masked, warped_B_masked

if __name__ == "__main__":
    main()