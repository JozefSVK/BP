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

def segment_head_placeholder(image_rgb):
    depth_model = load_depth_anything_model("vits")
    inverted_depth_map = get_depth_map(depth_model, image_rgb)
    mask = get_foreground_mask(inverted_depth_map, 71)
    plt.imshow(mask)
    plt.show()
    return mask
    """
    !!! PLACEHOLDER FUNKCIA PRE SEGMENTACIU HLAVY !!!
    Vykona velmi zakladnu segmentaciu pomocou prahovania.
    TOTO JE POTREBNE NAHRADIT ROBUSTNOU METODOU.
    """
    print("VAROVANIE: Pouziva sa PLACEHOLDER segmentacia. Vysledky mozu byt nepresne.")
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Jednoduche prahovanie - hodnotu treba potencialne upravit
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Volitelne: Morfologicke uzavretie na vyplnenie malych dier
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

def find_largest_contour(mask):
    """Najde najvacsiu vonkajsiu konturu v binarnej maske."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # CHAIN_APPROX_NONE pre viac bodov
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    # Vstup je Nx1x2, chceme Nx2 pre lahsie spracovanie
    return largest_contour.reshape(-1, 2)

def sample_contour(contour, num_points):
    """Navzorkuje priblizne rovnomerny pocet bodov z kontury."""
    if contour is None or len(contour) < 2:
        return np.array([])
    # Jednoduche vzorkovanie podla indexov
    indices = np.linspace(0, len(contour) - 1, num_points, dtype=int, endpoint=True)
    sampled_points = contour[indices]
    return sampled_points

def match_contour_points_stereo(points_A, contour_B, img_A_gray, img_B_gray, patch_size=7, y_tolerance=1):
    """
    Páruje body z points_A s bodmi na contour_B pomocou stereo obmedzenia a patch matchingu (SSD).
    """
    matched_A = []
    matched_B = []
    h, w = img_A_gray.shape
    half_patch = patch_size // 2

    # Pre rychlesie hladanie kandidatov podla Y suradnice
    contour_B_dict_y = {}
    for i, (bx, by) in enumerate(contour_B):
        if by not in contour_B_dict_y:
            contour_B_dict_y[by] = []
        contour_B_dict_y[by].append(i) # Ulozime index bodu v povodnej konture

    print(f"Párovanie {len(points_A)} bodov kontúry A...")
    start_time = time.time()

    for i, (xa, ya) in enumerate(points_A):
        candidates_indices = []
        # Hladaj kandidatov v tolerovanom pasme Y
        for y_check in range(ya - y_tolerance, ya + y_tolerance + 1):
            if y_check in contour_B_dict_y:
                candidates_indices.extend(contour_B_dict_y[y_check])

        if not candidates_indices:
            # print(f" Bod A {i} ({xa},{ya}): Nenasiel sa kandidat v riadku.")
            continue # Nenasiel sa ziaden kandidat v danom riadku

        # Ziskaj patch okolo bodu A
        # Osetrenie okrajov
        r_start_A = max(0, ya - half_patch)
        r_end_A = min(h, ya + half_patch + 1)
        c_start_A = max(0, xa - half_patch)
        c_end_A = min(w, xa + half_patch + 1)
        patch_A = img_A_gray[r_start_A:r_end_A, c_start_A:c_end_A]

        # Ak je patch prilis maly kvoli okraju, preskocime
        if patch_A.shape[0] != patch_size or patch_A.shape[1] != patch_size:
             # print(f" Bod A {i} ({xa},{ya}): Patch prilis maly.")
             continue

        best_match_idx = -1
        min_ssd = float('inf')

        # Hladaj najlepsieho kandidata pomocou SSD
        for idx_b in candidates_indices:
            xb, yb = contour_B[idx_b]

            # Ziskaj patch okolo kandidata B
            r_start_B = max(0, yb - half_patch)
            r_end_B = min(h, yb + half_patch + 1)
            c_start_B = max(0, xb - half_patch)
            c_end_B = min(w, xb + half_patch + 1)
            patch_B = img_B_gray[r_start_B:r_end_B, c_start_B:c_end_B]

            # Ak je patch prilis maly alebo nezhodnych rozmerov
            if patch_B.shape != patch_A.shape:
                continue

            ssd = np.sum((patch_A.astype(np.float32) - patch_B.astype(np.float32))**2)

            if ssd < min_ssd:
                min_ssd = ssd
                best_match_idx = idx_b

        if best_match_idx != -1:
            matched_A.append((xa, ya))
            matched_B.append(tuple(contour_B[best_match_idx]))
        # else:
             # print(f" Bod A {i} ({xa},{ya}): Nenasiel sa dobry match napriek kandidatom.")

    end_time = time.time()
    print(f"Párovanie dokončené za {end_time - start_time:.2f}s. Nájdených {len(matched_A)} párov kontúry.")
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
mask_A = segment_head_placeholder(img_A_rgb)
mask_B = segment_head_placeholder(img_B_rgb)

# 3. Extrakcia kontur
print("Hľadám kontúry hlavy...")
contour_A = find_largest_contour(mask_A)
contour_B = find_largest_contour(mask_B)

if contour_A is None or contour_B is None:
    print("Chyba: Nepodarilo sa najst konturu hlavy na jednom alebo oboch obrazkoch (skontroluj segmentaciu).")
    exit()

# 4. Vzorkovanie a párovanie bodov kontury
print(f"Vzorkujem {NUM_CONTOUR_POINTS} bodov z kontúry A...")
sampled_contour_A = sample_contour(contour_A, NUM_CONTOUR_POINTS)

# Párovanie navzorkovanych bodov A s bodmi na celej konture B
matched_contour_A, matched_contour_B = match_contour_points_stereo(
    sampled_contour_A,
    contour_B, # Hladame na celej konture B
    img_A_gray,
    img_B_gray,
    patch_size=PATCH_SIZE,
    y_tolerance=Y_TOLERANCE
)

if not matched_contour_A:
    print("Chyba: Nepodarilo sa nájsť žiadne korešpondujúce body na kontúrach.")
    exit()

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