import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay

# ====== 1. Detekcia bodov tváre pomocou MediaPipe ======
def get_face_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        h, w = image.shape[:2]
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks.append((x, y))
        return np.array(landmarks, dtype=np.int32)

# ====== 2. Triangulácia a morphovanie ======
def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def morph_triangle(img1, img2, img_morph, t1, t2, t, alpha=0.5):
    # Bounding rectangles
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by bounding box origin
    t1_rect = [(pt[0] - r1[0], pt[1] - r1[1]) for pt in t1]
    t2_rect = [(pt[0] - r2[0], pt[1] - r2[1]) for pt in t2]
    t_rect = [(pt[0] - r[0], pt[1] - r[1]) for pt in t]

    # Masks and cropped triangles
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    img1_crop = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    img2_crop = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

    warp_img1 = apply_affine_transform(img1_crop, t1_rect, t_rect, (r[2], r[3]))
    warp_img2 = apply_affine_transform(img2_crop, t2_rect, t_rect, (r[2], r[3]))

    img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2
    img_morph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img_morph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask

# ====== 3. Hlavná funkcia ======
def morph_faces(img1, img2, alpha=0.5):
    points1 = get_face_landmarks(img1)
    points2 = get_face_landmarks(img2)
    if points1 is None or points2 is None:
        print("Nepodarilo sa získať body.")
        return None

    points = (1 - alpha) * points1 + alpha * points2
    points = points.astype(np.int32)

    # Vytvoríme Delaunay trianguláciu nad stredovými bodmi
    tri = Delaunay(points)
    img_morph = np.zeros(img1.shape, dtype=np.float32)

    for tri_indices in tri.simplices:
        t1 = [points1[i] for i in tri_indices]
        t2 = [points2[i] for i in tri_indices]
        t = [points[i] for i in tri_indices]
        morph_triangle(img1, img2, img_morph, t1, t2, t, alpha)

    return cv2.convertScaleAbs(img_morph)

# ====== 4. Načítanie a spustenie ======
img1 = cv2.imread("dataset/1/1/0003.png")
img2 = cv2.imread("dataset/1/1/0002.png")

morphed_face = morph_faces(img1, img2, alpha=0.5)
if morphed_face is not None:
    cv2.imshow("Stredový pohľad tváre", morphed_face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
