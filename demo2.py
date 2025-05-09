import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import time # For basic timing
import os
import sys
import torch
import traceback

print("Required libraries: opencv-python, mediapipe, numpy, matplotlib, scipy")

# --- Konfigurácia ---
IMAGE_PATH_A = 'dataset/1/1/0003.png' # ! NASTAV CESTU K LAVEMU OBRAZKU
IMAGE_PATH_B = 'dataset/1/1/0002.png' # ! NASTAV CESTU K PRAVEMU OBRAZKU
NUM_CONTOUR_POINTS = 100       # Pocet bodov na vzorkovanie kontury
PATCH_SIZE = 9                 # Velkost okna pre patch matching (neparne cislo)
Y_TOLERANCE = 1                # Tolerancia pre y-ovu suradnicu pri hladani stereo zhody
SHOW_PLOTS = True             # Ci zobrazit vysledky pomocou matplotlib

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

# --- Pomocné Funkcie ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
DEFAULT_ENCODER = 'vits'
DEPTH_THRESHOLD_PERCENTILE = 70 # Adjusted based on inverted depth (higher=closer)

# --- Depth Anything V2 Model Loading ---
def load_depth_anything_model(encoder_type="vits"):
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
        plt.imshow(inverted_depth_map)
        plt.show()
        # Use percentile on the inverted map (higher values are closer)
        threshold_value = np.percentile(inverted_depth_map, 100 - percentile)
        threshold_value = 0.440
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

def get_face_landmarks(image_rgb, face_mesh_detector):
    """Detekuje body tvare pomocou MediaPipe Face Mesh."""
    results = face_mesh_detector.process(image_rgb)
    h, w, _ = image_rgb.shape
    face_points = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Ziskame len prvu detekovanu tvar
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                face_points.append((x, y))
            break # Spracujeme len prvu tvar
    return face_points

def basic_head_segmentation(image_rgb):
    """Záložná základná segmentácia"""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask

def segment_head_only(image_rgb, face_points):
    """
    Segmentuje len hlavu bez krku a ramien použitím bodov tváre ako referenčných bodov
    """
    # Zisti bounding box tváre
    face_points_array = np.array(face_points)
    
    if len(face_points_array) == 0:
        return None
    
    # Nájdi hranice tváře
    x_min, y_min = face_points_array.min(axis=0)
    x_max, y_max = face_points_array.max(axis=0)
    
    # Rozšír o 20% pre zahrnutie čela a bočných častí
    margin_x = int((x_max - x_min) * 0.2)
    margin_y_top = int((y_max - y_min) * 0.3)  # Väčší margin hore pre čelo
    margin_y_bottom = int((y_max - y_min) * 0.1)  # Menší margin dole
    
    # Vytvor bounding box pre hlavu
    head_x_min = max(0, x_min - margin_x)
    head_x_max = min(image_rgb.shape[1], x_max + margin_x)
    head_y_min = max(0, y_min - margin_y_top)
    head_y_max = min(image_rgb.shape[0], y_max + margin_y_bottom)
    
    # Vytvor masku len pre túto oblasť
    head_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    
    # Naplň eliptickú oblasť namiesto obdĺžnika
    center_x = (head_x_min + head_x_max) // 2
    center_y = (head_y_min + head_y_max) // 2
    width = head_x_max - head_x_min
    height = head_y_max - head_y_min
    
    cv2.ellipse(head_mask, (center_x, center_y), (width//2, height//2), 0, 0, 360, 255, -1)
    
    # Vizualizácia pre debugging
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.plot([head_x_min, head_x_max, head_x_max, head_x_min, head_x_min], 
             [head_y_min, head_y_min, head_y_max, head_y_max, head_y_min], 'r-')
    plt.title('Head Region Detection')
    plt.subplot(1, 2, 2)
    plt.imshow(head_mask, cmap='gray')
    plt.title('Head Mask')
    plt.show()
    
    return head_mask

# 2. Zlepšená funkcia na filtrovanie bodů kontúry len na hlavu
def filter_contour_to_head(contour, head_mask, face_points):
    """Filtruje body kontúry tak, aby patrili len ku hlave"""
    if contour is None or head_mask is None:
        return contour
    
    filtered_contour = []
    
    # Nájdi hranice hlavy z bodov tváre
    face_points_array = np.array(face_points)
    if len(face_points_array) > 0:
        y_max_face = face_points_array[:, 1].max()
        
        for point in contour:
            x, y = point
            # Skontroluj, či bod je v rámci masky
            if 0 <= x < head_mask.shape[1] and 0 <= y < head_mask.shape[0]:
                # Po invertovaní: 255 = hlava, 0 = pozadie
                if head_mask[y, x] > 0:  # Body kde je hlava
                    # Pridaj extra kontrolu pre spodnú hranicu (vylúč krk)
                    if y <= y_max_face + 50:  # 50 pixelov pod najnižší bod tváre
                        filtered_contour.append(point)
    
    return np.array(filtered_contour)


def segment_head_placeholder(image_rgb, face_points):
    """Vylepšená segmentácia hlavy použitím depth a referenčných bodov"""
    # Najprv skús depth-based segmentation
    depth_model = load_depth_anything_model("vits")
    inverted_depth_map = get_depth_map(depth_model, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    
    if inverted_depth_map is not None:
        threshold_value = .440
        mask = (inverted_depth_map > threshold_value).astype(np.uint8) * 255
        
        # Post-processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    else:
        mask = segment_head_only(image_rgb, face_points)
    
    # Intersection s head-only maskou
    head_only_mask = segment_head_only(image_rgb, face_points)
    if head_only_mask is not None:
        mask = cv2.bitwise_and(mask, head_only_mask)
    
    # További čistenie od krku
    if len(face_points) > 0:
        y_max_face = max(point[1] for point in face_points)
        
        # Postupné vyhladenie dole
        for y in range(y_max_face + 20, mask.shape[0]):
            alpha = 1.0 - min(1.0, (y - y_max_face - 20) / 50.0)
            if alpha <= 0:
                mask[y:, :] = 0  # Set to 0 (background), not 255
                break
            mask[y, :] = mask[y, :] * alpha
    
    return mask

def find_largest_contour(mask):
    """Najde najvacsiu vonkajsiu konturu v binarnej maske."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # CHAIN_APPROX_NONE pre viac bodov
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    # Vstup je Nx1x2, chceme Nx2 pre lahsie spracovanie
    return largest_contour.reshape(-1, 2)

# 2. Zlepšená function na zjednotenie kontúry
def unify_contour(mask):
    """Zjednotí fragmentovanú kontúru na jednu súvislú"""
    # Nájdi všetky kontúry
    contours, _ = cv2.findContours(cv2.bitwise_not(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return None
    
    # Kombinuj všetky kontúry do jednej
    combined_contour = []
    for cnt in contours:
        for point in cnt.reshape(-1, 2):
            combined_contour.append(point)
    
    if len(combined_contour) < 3:
        return None
        
    # Usporiadaj body v kruhovom poradí
    center = np.mean(combined_contour, axis=0)
    angles = np.arctan2([p[1] - center[1] for p in combined_contour], 
                        [p[0] - center[0] for p in combined_contour])
    sorted_indices = np.argsort(angles)
    
    sorted_contour = np.array([combined_contour[i] for i in sorted_indices])
    
    return sorted_contour

# 3. Upravený filter kontúry - méně restriktívní
def filter_contour_relaxed(contour, head_mask, face_points):
    """Mírnější filtrování kontúry"""
    if contour is None or head_mask is None:
        return contour
    
    filtered_contour = []
    
    # Nájdi hranice hlavy z bodov tváre
    face_points_array = np.array(face_points)
    if len(face_points_array) > 0:
        y_min_face = face_points_array[:, 1].min()
        y_max_face = face_points_array[:, 1].max()
        face_height = y_max_face - y_min_face
        
        for point in contour:
            x, y = point
            # Skontroluj hranice
            if 0 <= x < head_mask.shape[1] and 0 <= y < head_mask.shape[0]:
                # Prijmi body kde je hlava (255)
                if head_mask[y, x] > 0:
                    # Mírnější kontrola spodnej hranice 
                    if y <= y_max_face + int(face_height * 0.2):  # 20% výšky tváre navyše
                        filtered_contour.append(point)
    
    return np.array(filtered_contour)

def sample_contour(contour, num_points):
    """Sample fewer contour points and smooth them"""
    if contour is None or len(contour) < 2:
        return np.array([])
    
    # Sample fewer points
    indices = np.linspace(0, len(contour) - 1, num_points, dtype=int, endpoint=True)
    sampled_points = contour[indices]
    
    # Smooth the contour to avoid jagged edges
    # Apply Gaussian smoothing to the coordinates
    sigma = 2.0
    from scipy.ndimage import gaussian_filter1d
    
    # Close the contour to make it cyclic
    closed_contour = np.vstack([sampled_points, sampled_points[0]])
    
    # Smooth x and y coordinates separately
    smoothed_x = gaussian_filter1d(closed_contour[:, 0], sigma, mode='wrap')
    smoothed_y = gaussian_filter1d(closed_contour[:, 1], sigma, mode='wrap')
    
    # Remove the duplicate point we added
    smoothed_points = np.column_stack([smoothed_x[:-1], smoothed_y[:-1]])
    
    return smoothed_points

def add_boundary_points(points, img_shape):
    """Add boundary points around the image to improve triangulation"""
    h, w = img_shape[:2]
    boundary_points = []
    
    # Add corners
    boundary_points.extend([(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)])
    
    # Add edge points
    edge_step = 50  # Adjust based on needs
    for i in range(edge_step, w-1, edge_step):
        boundary_points.append((i, 0))
        boundary_points.append((i, h-1))
    for i in range(edge_step, h-1, edge_step):
        boundary_points.append((0, i))
        boundary_points.append((w-1, i))
    
    return np.vstack([points, np.array(boundary_points)])

def match_contours_bidirectional(contour_A, contour_B, img_A_gray, img_B_gray, num_points=80, patch_size=9, y_tolerance=2):
    """Bidirectional contour matching with better results"""
    
    # Vzorkuj obe kontúry
    sampled_A = sample_contour(contour_A, num_points)
    sampled_B = sample_contour(contour_B, num_points)
    
    # Vytvor mapovanie bodov
    matched_A = []
    matched_B = []
    
    h, w = img_A_gray.shape
    half_patch = patch_size // 2
    
    for i, (xa, ya) in enumerate(sampled_A):
        best_match_idx = -1
        min_ssd = float('inf')
        
        # Hľadaj v okolí y-ových pozícií
        for j, (xb, yb) in enumerate(sampled_B):
            if abs(ya - yb) > y_tolerance:
                continue
                
            # Zisti patch
            r_start_A = max(0, ya - half_patch)
            r_end_A = min(h, ya + half_patch + 1)
            c_start_A = max(0, xa - half_patch)
            c_end_A = min(w, xa + half_patch + 1)
            patch_A = img_A_gray[r_start_A:r_end_A, c_start_A:c_end_A]
            
            r_start_B = max(0, yb - half_patch)
            r_end_B = min(h, yb + half_patch + 1)
            c_start_B = max(0, xb - half_patch)
            c_end_B = min(w, xb + half_patch + 1)
            patch_B = img_B_gray[r_start_B:r_end_B, c_start_B:c_end_B]
            
            if patch_A.shape != patch_B.shape:
                continue
                
            # Počítaj SSD
            ssd = np.sum((patch_A.astype(np.float32) - patch_B.astype(np.float32))**2)
            
            if ssd < min_ssd:
                min_ssd = ssd
                best_match_idx = j
        
        # Prijmi match ak je SSD pod prahom
        if best_match_idx != -1 and min_ssd < 10000:  # Zvýš threshold
            matched_A.append((xa, ya))
            matched_B.append(tuple(sampled_B[best_match_idx]))
    
    return matched_A, matched_B

def apply_morph(img_shape, img_src, points_src, points_dst, delaunay_tri):
    """Aplikuje morfovanie (warp) jedneho obrazka na cielovu geometriu."""
    h, w, _ = img_shape
    warped_img = np.zeros(img_shape, dtype=img_src.dtype)

    for indices in delaunay_tri.simplices:
        # Ziskaj vrcholy trojuholnikov pre zdroj a ciel
        tri_src_pts = np.float32([points_src[i] for i in indices])
        tri_dst_pts = np.float32([points_dst[i] for i in indices])

        # Vypocitaj afinnu transformaciu
        M = cv2.getAffineTransform(tri_src_pts, tri_dst_pts)

        # Vytvor masku pre cielovy trojuholnik
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(tri_dst_pts), 255)

        # Warpuj cely zdrojovy obrazok
        # (efektivnejsie by bolo warpovat len bounding box, ale toto je jednoduchsie)
        img_warped_full = cv2.warpAffine(img_src, M, (w, h))

        # Skopiruj warpnutu cast len v ramci masky do vysledneho obrazku
        # Pouzitie masky zabezpeci, ze sa neprepisuju pixely z inych trojuholnikov
        warped_img = np.where(mask[:, :, None].astype(bool), img_warped_full, warped_img)

    return warped_img

def apply_morph_to_roi(img_shape, img_src, points_src, points_dst, delaunay_tri, roi_mask=None):
    """Apply morphing only within the region of interest"""
    h, w, _ = img_shape
    warped_img = np.zeros(img_shape, dtype=img_src.dtype)
    
    # Create a valid region mask based on triangulation
    valid_mask = np.zeros((h, w), dtype=np.uint8)
    
    for indices in delaunay_tri.simplices:
        tri_src_pts = np.float32([points_src[i] for i in indices])
        tri_dst_pts = np.float32([points_dst[i] for i in indices])
        
        # Calculate affine transformation
        M = cv2.getAffineTransform(tri_src_pts, tri_dst_pts)
        
        # Create mask for destination triangle
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(tri_dst_pts), 255)
        
        # Update valid region mask
        valid_mask = cv2.bitwise_or(valid_mask, mask)
        
        # Warp the image
        img_warped_full = cv2.warpAffine(img_src, M, (w, h))
        
        # Copy only the triangle region
        warped_img = np.where(mask[:, :, None].astype(bool), img_warped_full, warped_img)
    
    # If ROI mask is provided, only keep the warped region within the mask
    if roi_mask is not None:
        roi_mask_3ch = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2RGB)
        warped_img = cv2.bitwise_and(warped_img, roi_mask_3ch)
    
    return warped_img, valid_mask

# Pridaj dodatočné body pre lepšiu triangulation
def add_extra_contour_points(matched_A, matched_B, num_extra=20):
    """Pridať extra body medzi existujúce body"""
    extra_A = []
    extra_B = []
    
    for i in range(len(matched_A) - 1):
        # Interpoluj body medzi existujúcimi
        for j in range(1, num_extra // len(matched_A) + 1):
            alpha = j / (num_extra // len(matched_A) + 1)
            
            x_a = matched_A[i][0] * (1 - alpha) + matched_A[i + 1][0] * alpha
            y_a = matched_A[i][1] * (1 - alpha) + matched_A[i + 1][1] * alpha
            
            x_b = matched_B[i][0] * (1 - alpha) + matched_B[i + 1][0] * alpha
            y_b = matched_B[i][1] * (1 - alpha) + matched_B[i + 1][1] * alpha
            
            extra_A.append((x_a, y_a))
            extra_B.append((x_b, y_b))
    
    return matched_A + extra_A, matched_B + extra_B

# --- Hlavný Skript ---

# Nacitaj obrazky
img_A_bgr = cv2.imread(IMAGE_PATH_A)
img_B_bgr = cv2.imread(IMAGE_PATH_B)

if img_A_bgr is None or img_B_bgr is None:
    print(f"Chyba: Nepodarilo sa nacitat obrazky. Skontroluj cesty: '{IMAGE_PATH_A}', '{IMAGE_PATH_B}'")
    exit()

# Prevod do RGB pre MediaPipe a Matplotlib
img_A_rgb = cv2.cvtColor(img_A_bgr, cv2.COLOR_BGR2RGB)
img_B_rgb = cv2.cvtColor(img_B_bgr, cv2.COLOR_BGR2RGB)
img_A_gray = cv2.cvtColor(img_A_rgb, cv2.COLOR_RGB2GRAY)
img_B_gray = cv2.cvtColor(img_B_rgb, cv2.COLOR_RGB2GRAY)

h, w, _ = img_A_rgb.shape

# 1. Detekcia bodov tvare
print("Detekujem body tváre (MediaPipe)...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True, # Presnejsie body okolo oci a pier
    min_detection_confidence=0.5)

face_points_A = get_face_landmarks(img_A_rgb, face_mesh_detector)
face_points_B = get_face_landmarks(img_B_rgb, face_mesh_detector)

if not face_points_A or not face_points_B:
    print("Chyba: Nepodarilo sa detekovat tvar na jednom alebo oboch obrazkoch.")
    exit()
# MediaPipe dava viac bodov, ale mali by byt konzistentne indexovane
if len(face_points_A) != len(face_points_B):
     print("Varovanie: MediaPipe vratil rozny pocet bodov pre obrazky A a B. Pouzijem mensi pocet.")
     min_len = min(len(face_points_A), len(face_points_B))
     face_points_A = face_points_A[:min_len]
     face_points_B = face_points_B[:min_len]

print(f"Nájdených {len(face_points_A)} bodov tváre.")

# 2. Segmentacia hlavy (Placeholder)
print("Segmentujem hlavu (Placeholder)...")
mask_A = segment_head_placeholder(img_A_rgb, face_points_A)
mask_B = segment_head_placeholder(img_B_rgb, face_points_B)

# Vizualizuj masky
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(mask_A, cmap='gray')
plt.title('Maska A - finálna')
plt.subplot(1, 2, 2)
plt.imshow(mask_B, cmap='gray')
plt.title('Maska B - finálna')
plt.show()

# 3. Extrakcia kontur
print("Hľadám kontúry hlavy...")
contour_A = unify_contour(mask_A)
contour_B = unify_contour(mask_B)

if contour_A is None or contour_B is None:
    print("Chyba: Nepodarilo sa najst konturu hlavy na jednom alebo oboch obrazkoch (skontroluj segmentaciu).")
    exit()

# Filtruj kontury
# contour_A_filtered = filter_contour_relaxed(contour_A, mask_A, face_points_A)
# contour_B_filtered = filter_contour_relaxed(contour_B, mask_B, face_points_B)
contour_A_filtered = contour_A
contour_B_filtered = contour_B

# Vizualizuj filtrované kontury
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_A_rgb)
if len(contour_A_filtered) > 0:
    plt.plot(contour_A_filtered[:, 0], contour_A_filtered[:, 1], 'r-', linewidth=2)
    plt.scatter(contour_A_filtered[:, 0], contour_A_filtered[:, 1], c='r', s=5)
plt.title(f'Kontúra A - {len(contour_A_filtered)} bodov')

plt.subplot(1, 2, 2)
plt.imshow(img_B_rgb)
if len(contour_B_filtered) > 0:
    plt.plot(contour_B_filtered[:, 0], contour_B_filtered[:, 1], 'g-', linewidth=2)
    plt.scatter(contour_B_filtered[:, 0], contour_B_filtered[:, 1], c='g', s=5)
plt.title(f'Kontúra B - {len(contour_B_filtered)} bodov')
plt.show()

# Skontroluj či máme dosť bodov na matching
if len(contour_A_filtered) < 20 or len(contour_B_filtered) < 20:
    print(f"Varovanie: Málo bodov kontúry - A: {len(contour_A_filtered)}, B: {len(contour_B_filtered)}")
    
    # Ak je málo bodov, použi celé kontury bez filtrovania
    print("Používam nefiltrované kontury...")
    contour_A_filtered = contour_A
    contour_B_filtered = contour_B

# Pokračuj s vzorkovaním
if len(contour_A_filtered) > 0 and len(contour_B_filtered) > 0:
    sampled_contour_A = sample_contour(contour_A_filtered, NUM_CONTOUR_POINTS)
    sampled_contour_B = sample_contour(contour_B_filtered, NUM_CONTOUR_POINTS)
    
    # Bidirectional matching
    matched_contour_A, matched_contour_B = match_contours_bidirectional(
        contour_A_filtered,
        contour_B_filtered,
        img_A_gray,
        img_B_gray,
        num_points=NUM_CONTOUR_POINTS,
        patch_size=PATCH_SIZE,
        y_tolerance=Y_TOLERANCE
    )
else:
    print("Chyba: Žiadne body kontúry po filtrovaní!")
    exit()

if not matched_contour_A:
    print("Chyba: Nepodarilo sa nájsť žiadne korešpondujúce body na kontúrach.")
    exit()
else:
     matched_contour_A, matched_contour_B = add_extra_contour_points(
        matched_contour_A, matched_contour_B, num_extra=40)
     print(f"Celkom {len(matched_contour_A)} bodov po interpolácii")

# 5. Kombinacia bodov (Tvár + Kontúra)
# Prevod na numpy polia pre lahsie indexovanie
face_points_A_np = np.array(face_points_A, dtype=np.float32)
face_points_B_np = np.array(face_points_B, dtype=np.float32)
contour_points_A_np = np.array(matched_contour_A, dtype=np.float32)
contour_points_B_np = np.array(matched_contour_B, dtype=np.float32)

# Spojenie bodov
all_points_A = np.vstack((face_points_A_np, contour_points_A_np))
all_points_B = np.vstack((face_points_B_np, contour_points_B_np))

print(f"Celkový počet korešpondujúcich bodov pre morfovanie: {len(all_points_A)}")

# 6. Morfovanie
print("Vykonávam morfovanie...")
start_morph_time = time.time()

# Vypocet stredovych bodov
mid_points = (all_points_A + all_points_B) * 0.5

# Triangulacia stredovych bodov
# Pouzijeme SciPy Delaunay, lebo vracia indexy, co ulahcuje mapovanie
try:
    tri = Delaunay(mid_points)
    print(f"Vytvorených {len(tri.simplices)} trojuholníkov pre morfovanie.")
except Exception as e:
    print(f"Chyba pri Delaunay triangulácii: {e}")
    print("Skontrolujte, či nemáte duplicitné alebo kolineárne body.")
    exit()


# Warpnutie obrazkov na stredovu geometriu
warped_A = apply_morph((h, w, 3), img_A_rgb, all_points_A, mid_points, tri)
warped_B = apply_morph((h, w, 3), img_B_rgb, all_points_B, mid_points, tri)

# Prelinanie warpnutych obrazkov
morphed_image = cv2.addWeighted(warped_A, 0.5, warped_B, 0.5, 0)

end_morph_time = time.time()
print(f"Morfovanie dokončené za {end_morph_time - start_morph_time:.2f}s.")

# 7. Zobrazenie vysledkov (ak je povolene)
if SHOW_PLOTS:
    print("Zobrazujem výsledky...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Povodny obrazok A s bodmi
    axes[0, 0].imshow(img_A_rgb)
    axes[0, 0].scatter(face_points_A_np[:, 0], face_points_A_np[:, 1], s=5, c='r', label='Tvár A')
    axes[0, 0].scatter(contour_points_A_np[:, 0], contour_points_A_np[:, 1], s=5, c='b', label='Kontúra A')
    axes[0, 0].set_title('Obrázok A + Body')
    axes[0, 0].legend()
    axes[0, 0].axis('off')

    # Povodny obrazok B s bodmi
    axes[0, 1].imshow(img_B_rgb)
    axes[0, 1].scatter(face_points_B_np[:, 0], face_points_B_np[:, 1], s=5, c='r', label='Tvár B')
    axes[0, 1].scatter(contour_points_B_np[:, 0], contour_points_B_np[:, 1], s=5, c='lime', label='Kontúra B')
    axes[0, 1].set_title('Obrázok B + Body')
    axes[0, 1].legend()
    axes[0, 1].axis('off')

    # Morfovany obrazok so stredovymi bodmi
    axes[0, 2].imshow(morphed_image)
    axes[0, 2].scatter(mid_points[:, 0], mid_points[:, 1], s=3, c='yellow', alpha=0.5)
    axes[0, 2].set_title('Výsledný Stredový Pohľad (Morfovaný)')
    axes[0, 2].axis('off')

    # Maska A
    axes[1, 0].imshow(mask_A, cmap='gray')
    axes[1, 0].set_title('Segmentačná Maska A (Placeholder)')
    axes[1, 0].axis('off')

    # Maska B
    axes[1, 1].imshow(mask_B, cmap='gray')
    axes[1, 1].set_title('Segmentačná Maska B (Placeholder)')
    axes[1, 1].axis('off')

    # Warpnuty A
    axes[1, 2].imshow(warped_A)
    axes[1, 2].set_title('Warpnutý A na Stred')
    axes[1, 2].axis('off')


    plt.tight_layout()
    plt.show()

print("Skript dokončený.")