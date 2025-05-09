import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import time
import os
import torch # Potrebné pre Transformers

# <<< PRIDANE: Importy pre Transformers (Hugging Face)
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

print("Required libraries: opencv-python, mediapipe, numpy, matplotlib, scipy, transformers, torch, accelerate")
print("UPOZORNENIE: Používa sa Depth Anything V2 cez Hugging Face Transformers.")

# --- Konfigurácia ---
IMAGE_PATH_A = 'dataset/1/1/0003.png' # ! NASTAV CESTU K LAVEMU OBRAZKU
IMAGE_PATH_B = 'dataset/1/1/0002.png' # ! NASTAV CESTU K PRAVEMU OBRAZKU

# <<< ZMENA: Názov modelu Depth Anything V2 na Hugging Face Hub
# Môžeš vyskúšať aj 'LiheYoung/depth-anything-v2-base' alebo 'LiheYoung/depth-anything-v2-large' (náročnejšie)
DEPTH_MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

# ! TENTO PRAH BUDEŠ MUSIEŤ STÁLE LADIT PODĽA VÝSLEDKOV !
DEPTH_THRESHOLD_VALUE = 150  # <<<<< EXPERIMENTUJ S TOUTO HODNOTOU (0-255)
# Predpokladáme, že po normalizácii nižšie hodnoty znamenajú bližšie
# Ak je to naopak, zmeň na False
ASSUME_LOWER_IS_CLOSER = True

NUM_CONTOUR_POINTS = 160
PATCH_SIZE = 9
Y_TOLERANCE = 1
SHOW_PLOTS = True

# --- Detekcia Zariadenia (GPU alebo CPU) ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Používa sa zariadenie: {device}")

# --- Inicializácia MediaPipe a Depth Modelov ---
print("Inicializujem MediaPipe a Depth modely...")

# MediaPipe Face Mesh (stále zo solutions API pre jednoduchosť)
mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# <<< ZMENA: Inicializácia Depth Anything V2 pomocou Transformers
try:
    depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_NAME)
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_NAME).to(device)
    print(f"Model '{DEPTH_MODEL_NAME}' načítaný na {device}.")
except Exception as e:
    print(f"CHYBA pri načítavaní modelu Depth Anything V2: {e}")
    print("Skontrolujte názov modelu a internetové pripojenie.")
    if 'face_mesh_detector' in locals(): face_mesh_detector.close()
    exit()


# --- Pomocné Funkcie ---

def get_face_landmarks(image_rgb, face_mesh_detector):
    results = face_mesh_detector.process(image_rgb)
    h, w, _ = image_rgb.shape
    face_points = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                face_points.append((x, y))
            break
    return face_points

# <<< ZMENA: Funkcia pre segmentáciu pomocou Depth Anything V2
def segment_head_depth_anything(image_rgb, processor, model, device, threshold, assume_lower_is_closer=True):
    """
    Vykoná segmentáciu pomocou Depth Anything V2 a prahovania.
    Vráti binárnu masku (0 alebo 255).
    """
    h_orig, w_orig, _ = image_rgb.shape

    # Priprav obrazok pre model
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)

    # Ziskaj predikciu hĺbky
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpoluj na povodnu velkost
    # Pozor: výstup môže mať iné rozmery ako vstup
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(h_orig, w_orig),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Prevod na numpy a normalizacia na 0-255 pre lahšie prahovanie
    # Raw výstup je relatívna hĺbka, normalizácia pomôže konzistencii prahu
    depth_map = prediction.cpu().numpy()
    normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # --- Aplikácia prahovania ---
    if assume_lower_is_closer:
        # Nižšie hodnoty ako prah budú biele (popredie)
        threshold_type = cv2.THRESH_BINARY_INV
    else:
        # Vyššie hodnoty ako prah budú biele (popredie)
        threshold_type = cv2.THRESH_BINARY

    _, mask = cv2.threshold(normalized_depth_map,
                             threshold,
                             255,
                             threshold_type)

    # --- Voliteľný Post-processing Masky ---
    # kernel_size = 5
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask, normalized_depth_map # Vrátime aj normalizovanú mapu pre zobrazenie

# --- Funkcie find_largest_contour, sample_contour, match_contour_points_stereo, apply_morph zostávajú rovnaké ---
# ... (ich kód je rovnaký ako v predchádzajúcej verzii) ...
def find_largest_contour(mask, padding=1):
    if mask is None:
        return None

    # Získaj rozmery pôvodnej masky
    h, w = mask.shape[:2]

    # Vytvor väčšiu čiernu plochu (s paddingom)
    padded_h = h + 2 * padding
    padded_w = w + 2 * padding
    padded_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)

    # Skopíruj pôvodnú masku do stredu novej plochy
    padded_mask[padding:padding + h, padding:padding + w] = mask

    # plt.imshow(padded_mask)
    # plt.show()

    # Nájdi kontúry na maske s paddingom
    contours, _ = cv2.findContours(padded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(f"DEBUG (Padding): Nájdených {len(contours)} externých kontúr.") # Ponechaj pre debug
    # plt.imshow(padded_mask)
    # for i, c in enumerate(contours):
    #     points = c.reshape(-1, 2)
    #     # Pouzi rozne farby/znacky pre lepsiu identifikaciu
    #     plt.plot(points[:, 0], points[:, 1], '.', markersize=2, label=f'Nájdená Kont. B {i+1}')
    # plt.show()

    if not contours:
        print("DEBUG (Padding): Žiadne kontúry nenájdené na maske s paddingom.")
        return None

    # Nájdi najväčšiu kontúru podľa plochy
    largest_contour_padded = max(contours, key=cv2.contourArea)

    # !!! DÔLEŽITÉ: Posuň súradnice kontúry späť o padding !!!
    # Keďže sme kontúru našli na väčšej ploche, musíme odpočítať padding,
    # aby súradnice zodpovedali pôvodnému obrázku.
    largest_contour_original_coords = largest_contour_padded.reshape(-1, 2) - padding

    return largest_contour_original_coords

def sample_contour(contour, num_points):
    if contour is None or len(contour) < 2: return np.array([])
    indices = np.linspace(0, len(contour) - 1, num_points, dtype=int, endpoint=True)
    return contour[indices]

def match_contour_points_stereo(points_A, contour_B, img_A_gray, img_B_gray, patch_size=7, y_tolerance=1):
    matched_A, matched_B = [], []
    h, w = img_A_gray.shape
    half_patch = patch_size // 2
    contour_B_dict_y = {}
    for i, (bx, by) in enumerate(contour_B):
        if by not in contour_B_dict_y: contour_B_dict_y[by] = []
        contour_B_dict_y[by].append(i)
    # print(f"Párovanie {len(points_A)} bodov kontúry A...")
    # start_time = time.time()
    for i, (xa, ya) in enumerate(points_A):
        candidates_indices = []
        for y_check in range(ya - y_tolerance, ya + y_tolerance + 1):
            if y_check in contour_B_dict_y: candidates_indices.extend(contour_B_dict_y[y_check])
        if not candidates_indices: continue
        r_start_A,r_end_A = max(0, ya - half_patch),min(h, ya + half_patch + 1)
        c_start_A,c_end_A = max(0, xa - half_patch),min(w, xa + half_patch + 1)
        patch_A = img_A_gray[r_start_A:r_end_A, c_start_A:c_end_A]
        if patch_A.shape[0]!= patch_size or patch_A.shape[1]!= patch_size: continue
        best_match_idx, min_ssd = -1, float('inf')
        for idx_b in candidates_indices:
            xb, yb = contour_B[idx_b]
            r_start_B,r_end_B = max(0, yb - half_patch),min(h, yb + half_patch + 1)
            c_start_B,c_end_B = max(0, xb - half_patch),min(w, xb + half_patch + 1)
            patch_B = img_B_gray[r_start_B:r_end_B, c_start_B:c_end_B]
            if patch_B.shape != patch_A.shape: continue
            ssd = np.sum((patch_A.astype(np.float32) - patch_B.astype(np.float32))**2)
            if ssd < min_ssd: min_ssd, best_match_idx = ssd, idx_b
        if best_match_idx != -1:
            matched_A.append((xa, ya)); matched_B.append(tuple(contour_B[best_match_idx]))
    # end_time = time.time()
    print(f"Párovanie kontúr: Nájdených {len(matched_A)} párov.")
    return matched_A, matched_B

def match_stereo_points(points_ref, contour_target, img_ref_gray, img_target_gray,
                         patch_size=7, y_tolerance=1, description=""):
    """
    Páruje body z points_ref s bodmi na contour_target pomocou stereo obmedzenia a patch matchingu (SSD).

    Args:
        points_ref (np.array): Pole bodov (Nx2) z referenčného obrázka.
        contour_target (np.array): Kontúra (Mx2) z cieľového obrázka, na ktorej sa hľadá zhoda.
        img_ref_gray (np.array): Referenčný obrázok v odtieňoch sivej.
        img_target_gray (np.array): Cieľový obrázok v odtieňoch sivej.
        patch_size (int): Veľkosť okna pre porovnávanie.
        y_tolerance (int): Tolerancia y-ovej súradnice.
        description (str): Popis pre logovanie (napr. "A->B").

    Returns:
        tuple(list, list): Dvojica zoznamov (matched_ref_points, matched_target_points)
    """
    matched_ref_points = []
    matched_target_points = []
    h, w = img_ref_gray.shape
    half_patch = patch_size // 2

    # Pre rýchlejšie hľadanie kandidátov podľa Y súradnice v cieli
    contour_target_dict_y = {}
    for i, (tx, ty) in enumerate(contour_target):
        if ty not in contour_target_dict_y:
            contour_target_dict_y[ty] = []
        contour_target_dict_y[ty].append(i) # Uložíme index bodu v cieľovej kontúre

    print(f"Párovanie {description}: {len(points_ref)} referenčných bodov...")
    start_time = time.time()

    for i, (rx, ry) in enumerate(points_ref):
        candidates_indices = []
        # Hľadaj kandidátov v tolerovanom pásme Y v cieli
        for y_check in range(ry - y_tolerance, ry + y_tolerance + 1):
            if y_check in contour_target_dict_y:
                candidates_indices.extend(contour_target_dict_y[y_check])

        if not candidates_indices:
            continue # Nenašiel sa žiadny kandidát v danom riadku cieľa

        # Získaj patch okolo referenčného bodu
        r_start_ref = max(0, ry - half_patch)
        r_end_ref = min(h, ry + half_patch + 1)
        c_start_ref = max(0, rx - half_patch)
        c_end_ref = min(w, rx + half_patch + 1)
        patch_ref = img_ref_gray[r_start_ref:r_end_ref, c_start_ref:c_end_ref]

        if patch_ref.shape[0] != patch_size or patch_ref.shape[1] != patch_size:
             continue # Patch na okraji

        best_match_idx = -1
        min_ssd = float('inf')

        # Hľadaj najlepšieho kandidáta v cieli pomocou SSD
        for idx_target in candidates_indices:
            tx, ty = contour_target[idx_target]

            # Získaj patch okolo cieľového kandidáta
            r_start_target = max(0, ty - half_patch)
            r_end_target = min(h, ty + half_patch + 1)
            c_start_target = max(0, tx - half_patch)
            c_end_target = min(w, tx + half_patch + 1)
            patch_target = img_target_gray[r_start_target:r_end_target, c_start_target:c_end_target]

            if patch_target.shape != patch_ref.shape:
                continue # Rozdielne rozmery patchov (okraj)

            ssd = np.sum((patch_ref.astype(np.float32) - patch_target.astype(np.float32))**2)

            if ssd < min_ssd:
                min_ssd = ssd
                best_match_idx = idx_target

        if best_match_idx != -1:
            matched_ref_points.append((rx, ry))
            matched_target_points.append(tuple(contour_target[best_match_idx]))

    end_time = time.time()
    print(f"Párovanie {description} dokončené za {end_time - start_time:.2f}s. Nájdených {len(matched_ref_points)} párov.")
    return matched_ref_points, matched_target_points

def apply_morph(img_shape, img_src, points_src, points_dst, delaunay_tri):
    h, w, _ = img_shape
    warped_img = np.zeros(img_shape, dtype=img_src.dtype)
    for indices in delaunay_tri.simplices:
        tri_src_pts = np.float32([points_src[i] for i in indices])
        tri_dst_pts = np.float32([points_dst[i] for i in indices])
        M = cv2.getAffineTransform(tri_src_pts, tri_dst_pts)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(tri_dst_pts), 255)
        img_warped_full = cv2.warpAffine(img_src, M, (w, h))
        warped_img = np.where(mask[:, :, None].astype(bool), img_warped_full, warped_img)
    return warped_img

# --- Hlavný Skript ---

if __name__ == "__main__":
    # Nacitaj obrazky
    img_A_bgr = cv2.imread(IMAGE_PATH_A)
    img_B_bgr = cv2.imread(IMAGE_PATH_B)
    if img_A_bgr is None or img_B_bgr is None:
        print(f"Chyba: Nepodarilo sa nacitat obrazky.")
        if 'face_mesh_detector' in locals(): face_mesh_detector.close()
        # Nie je potreba .close() pre Transformers modely explicitne tu
        exit()

    img_A_rgb = cv2.cvtColor(img_A_bgr, cv2.COLOR_BGR2RGB)
    img_B_rgb = cv2.cvtColor(img_B_bgr, cv2.COLOR_BGR2RGB)
    img_A_gray = cv2.cvtColor(img_A_rgb, cv2.COLOR_RGB2GRAY)
    img_B_gray = cv2.cvtColor(img_B_rgb, cv2.COLOR_RGB2GRAY)
    h, w, _ = img_A_rgb.shape

    # 1. Detekcia bodov tvare
    print("Detekujem body tváre (MediaPipe Face Mesh)...")
    face_points_A = get_face_landmarks(img_A_rgb, face_mesh_detector)
    face_points_B = get_face_landmarks(img_B_rgb, face_mesh_detector)
    if not face_points_A or not face_points_B:
        print("Chyba: Nepodarilo sa detekovat tvar.")
        face_mesh_detector.close()
        exit()
    if len(face_points_A) != len(face_points_B):
        print("Varovanie: Rôzny počet bodov tváre. Použijem menší počet.")
        min_len = min(len(face_points_A), len(face_points_B))
        face_points_A, face_points_B = face_points_A[:min_len], face_points_B[:min_len]
    print(f"Nájdených {len(face_points_A)} bodov tváre.")

    # 2. Segmentacia pomocou Depth Anything V2
    print("Segmentujem pomocou Depth Anything V2...")
    start_depth_time = time.time()
    mask_A, depth_map_A_norm = segment_head_depth_anything(img_A_rgb, depth_processor, depth_model, device, DEPTH_THRESHOLD_VALUE, ASSUME_LOWER_IS_CLOSER)
    mask_B, depth_map_B_norm = segment_head_depth_anything(img_B_rgb, depth_processor, depth_model, device, DEPTH_THRESHOLD_VALUE, ASSUME_LOWER_IS_CLOSER)
    end_depth_time = time.time()
    print(f"Odhad hĺbky a prahovanie dokončené za {end_depth_time - start_depth_time:.2f}s.")

    if mask_A is None or mask_B is None: # Funkcia vracia None pri chybe
        print("Chyba: Segmentácia pomocou hĺbky zlyhala.")
        face_mesh_detector.close()
        exit()

    # plt.imshow(mask_A)
    # plt.show()
    # plt.imshow(mask_B)
    # plt.show()

    # 3. Extrakcia kontur
    print("Hľadám kontúry segmentov...")
    contour_A = find_largest_contour(mask_A)
    contour_B = find_largest_contour(mask_B)
    if contour_A is None or contour_B is None:
        print("Chyba: Nepodarilo sa najst konturu (skontroluj prah a kvalitu hĺbky).")
        face_mesh_detector.close()
        exit()

    # # 4. Vzorkovanie a párovanie bodov kontury
    # print(f"Vzorkujem {NUM_CONTOUR_POINTS} bodov z kontúry A...")
    # sampled_contour_A = sample_contour(contour_A, NUM_CONTOUR_POINTS)
    # matched_contour_A, matched_contour_B = match_contour_points_stereo(
    #     sampled_contour_A, contour_B, img_A_gray, img_B_gray,
    #     patch_size=PATCH_SIZE, y_tolerance=Y_TOLERANCE
    # )
    # if not matched_contour_A:
    #     print("Chyba: Nepodarilo sa nájsť korešpondujúce body na kontúrach.")
    #     face_mesh_detector.close()
    #     exit()

    # 4. Vzorkovanie a OBOJSMERNÉ párovanie bodov kontúry
print("--- Obojsmerné párovanie kontúr ---")

# 4a. Smer A -> B
print(f"Vzorkujem {NUM_CONTOUR_POINTS} bodov z kontúry A...")
sampled_contour_A = sample_contour(contour_A, NUM_CONTOUR_POINTS)
matched_A_from_A, matched_B_from_A = match_stereo_points(
    sampled_contour_A, contour_B, img_A_gray, img_B_gray,
    patch_size=PATCH_SIZE, y_tolerance=Y_TOLERANCE, description="A->B"
)

# 4b. Smer B -> A
print(f"Vzorkujem {NUM_CONTOUR_POINTS} bodov z kontúry B...")
sampled_contour_B = sample_contour(contour_B, NUM_CONTOUR_POINTS)
# Pozor na poradie argumentov! Referencia je B, cieľ je A.
# Výstup bude (zhody_na_A_najdene_z_B, body_z_B_ktore_boli_zhodou)
matched_B_from_B, matched_A_from_B = match_stereo_points(
    sampled_contour_B, contour_A, img_B_gray, img_A_gray, # Obrázky sú prehodené
    patch_size=PATCH_SIZE, y_tolerance=Y_TOLERANCE, description="B->A"
)

# 4c. Zlúčenie výsledkov (jednoduchá stratégia)
print("Zlučujem výsledky párovania...")
final_matched_A = list(matched_A_from_A) # Začni s výsledkami A->B
final_matched_B = list(matched_B_from_A)

# Použi set pre rýchlu kontrolu už pridaných bodov z B
# Prevod na tuple, aby boli hashable pre set
used_B_points = {tuple(p) for p in final_matched_B}

added_from_B_to_A = 0
# Prejdi výsledky B->A
# Pár je (zhoda_na_A, pôvodný_bod_z_B)
for point_A, point_B in zip(matched_A_from_B, matched_B_from_B):
    point_B_tuple = tuple(point_B)
    # Ak tento bod z B ešte nebol pridaný z párovania A->B
    if point_B_tuple not in used_B_points:
        final_matched_A.append(point_A)
        final_matched_B.append(point_B)
        used_B_points.add(point_B_tuple) # Označ ho ako použitý
        added_from_B_to_A += 1

print(f"Pridaných {added_from_B_to_A} unikátnych párov z párovania B->A.")

if not final_matched_A: # Skontroluj, či máme vôbec nejaké body
    print("Chyba: Po zlúčení nezostali žiadne korešpondujúce body na kontúrach.")
    face_mesh_detector.close()
    exit()
else:
    # 5. Kombinacia bodov
    face_points_A_np = np.array(face_points_A, dtype=np.float32)
    face_points_B_np = np.array(face_points_B, dtype=np.float32)
    # contour_points_A_np = np.array(matched_contour_A, dtype=np.float32)
    # contour_points_B_np = np.array(matched_contour_B, dtype=np.float32)
    # all_points_A = np.vstack((face_points_A_np, contour_points_A_np))
    # all_points_B = np.vstack((face_points_B_np, contour_points_B_np))
    # <<< ZMENA: Použitie finálnych zlúčených zoznamov
    contour_points_A_np = np.array(final_matched_A, dtype=np.float32)
    contour_points_B_np = np.array(final_matched_B, dtype=np.float32)
    # Skontroluj, či polia nie sú prázdne pred vstack
    if contour_points_A_np.size == 0:
        print("Varovanie: Nezostali žiadne body kontúry A po zlúčení.")
        all_points_A = face_points_A_np
    else:
        all_points_A = np.vstack((face_points_A_np, contour_points_A_np))

    if contour_points_B_np.size == 0:
        print("Varovanie: Nezostali žiadne body kontúry B po zlúčení.")
        all_points_B = face_points_B_np
    else:
        all_points_B = np.vstack((face_points_B_np, contour_points_B_np))

    print(f"Celkový počet korešpondujúcich bodov pre morfovanie: {len(all_points_A)}")
    # Skontroluj, či mame rovnaky pocet bodov v A a B (mali by sme mat)
    if len(all_points_A) != len(all_points_B):
        print("CHYBA: Nesedí počet bodov v A a B po zlúčení! Niekde je problém v logike.")
        face_mesh_detector.close(); exit()

    # 6. Morfovanie
    print("Vykonávam morfovanie...")
    start_morph_time = time.time()
    mid_points = (all_points_A + all_points_B) * 0.5
    try:
        tri = Delaunay(mid_points)
        print(f"Vytvorených {len(tri.simplices)} trojuholníkov pre morfovanie.")
    except Exception as e:
        print(f"Chyba pri Delaunay triangulácii: {e}")
        face_mesh_detector.close()
        exit()
    warped_A = apply_morph((h, w, 3), img_A_rgb, all_points_A, mid_points, tri)
    warped_B = apply_morph((h, w, 3), img_B_rgb, all_points_B, mid_points, tri)
    morphed_image = cv2.addWeighted(warped_A, 0.5, warped_B, 0.5, 0)
    end_morph_time = time.time()
    print(f"Morfovanie dokončené za {end_morph_time - start_morph_time:.2f}s.")

    plt.imshow(warped_A)
    plt.show()
    plt.imshow(warped_B)
    plt.show()

    # 7. Zobrazenie vysledkov
    if SHOW_PLOTS:
        print("Zobrazujem výsledky...")
        # <<< ZMENA: Zobrazime 3 riadky - originaly, hlbky/masky, vysledky
        fig, axes = plt.subplots(3, 3, figsize=(18, 15)) # Väčší figsize

        # Riadok 1: Originaly + Body
        axes[0, 0].imshow(img_A_rgb); axes[0, 0].scatter(face_points_A_np[:, 0], face_points_A_np[:, 1], s=5, c='r', label='Tvár A'); axes[0, 0].scatter(contour_points_A_np[:, 0], contour_points_A_np[:, 1], s=5, c='b', label='Kontúra A'); axes[0, 0].set_title('Obrázok A + Body'); axes[0, 0].legend(); axes[0, 0].axis('off')
        axes[0, 1].imshow(img_B_rgb); axes[0, 1].scatter(face_points_B_np[:, 0], face_points_B_np[:, 1], s=5, c='r', label='Tvár B'); axes[0, 1].scatter(contour_points_B_np[:, 0], contour_points_B_np[:, 1], s=5, c='lime', label='Kontúra B'); axes[0, 1].set_title('Obrázok B + Body'); axes[0, 1].legend(); axes[0, 1].axis('off')
        axes[0, 2].axis('off') # Prazdne miesto

        # temp_contours_B, _ = cv2.findContours(mask_B, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print(f"DEBUG PLOT: Kreslím {len(temp_contours_B)} nájdených kontúr pre B.")
        # for i, c in enumerate(temp_contours_B):
        #     points = c.reshape(-1, 2)
        #     # Pouzi rozne farby/znacky pre lepsiu identifikaciu
        #     axes[0, 1].plot(points[:, 0], points[:, 1], '.', markersize=2, label=f'Nájdená Kont. B {i+1}')
        # if temp_contours_B: # Ak sme nejake nasli, obnov legendu
        #     axes[0, 1].legend()

        # Riadok 2: Hlbkove mapy a masky
        axes[1, 0].imshow(depth_map_A_norm, cmap='plasma'); axes[1, 0].set_title('Normalizovaná Hĺbka A'); axes[1, 0].axis('off')
        axes[1, 1].imshow(depth_map_B_norm, cmap='plasma'); axes[1, 1].set_title('Normalizovaná Hĺbka B'); axes[1, 1].axis('off')
        # Zobrazime jednu z masiek (napr. A)
        axes[1, 2].imshow(mask_A, cmap='gray'); axes[1, 2].set_title(f'Maska A (Prah={DEPTH_THRESHOLD_VALUE})'); axes[1, 2].axis('off')


        # Riadok 3: Výsledky morfovania
        axes[2, 0].imshow(warped_A); axes[2, 0].set_title('Warpnutý A na Stred'); axes[2, 0].axis('off')
        axes[2, 1].imshow(warped_B); axes[2, 1].set_title('Warpnutý B na Stred'); axes[2, 1].axis('off') # <<< PRIDANE: Zobrazenie warp B
        axes[2, 2].imshow(morphed_image); axes[2, 2].scatter(mid_points[:, 0], mid_points[:, 1], s=3, c='yellow', alpha=0.5); axes[2, 2].set_title('Výsledný Stredový Pohľad'); axes[2, 2].axis('off')

        plt.tight_layout()
        plt.show()

    # Uvolnenie zdrojov MediaPipe
    face_mesh_detector.close()
    print("MediaPipe Face Mesh zdroje uvoľnené.")
    print("Skript dokončený.")