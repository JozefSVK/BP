import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# --------------------------
# 1) Face Mesh landmarks (478 bodov)
# --------------------------
def get_face_landmarks(image):
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5) as fm:
        res = fm.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    h, w = image.shape[:2]
    pts = []
    for lm in res.multi_face_landmarks[0].landmark:
        pts.append((int(lm.x*w), int(lm.y*h)))
    return np.array(pts, dtype=np.int32)  # (478, 2)

# --------------------------
# 2) Head silhouette contour extraction
# --------------------------
def get_head_contour(image, n_points=200):
    # Selfie segmentation
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        res = seg.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask = (res.segmentation_mask > 0.5).astype(np.uint8)*255
    # Find largest contour
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=lambda c: cv2.contourArea(c))
    cnt = cnt[:,0,:]  # shape (L,2)
    # Uniform sampling to n_points
    L = len(cnt)
    idx = np.round(np.linspace(0, L-1, n_points)).astype(int)
    sampled = cnt[idx]
    return sampled  # (n_points, 2)

# --------------------------
# 3) Visual sanity check
# --------------------------
def show_overlay(image, pts, title="Overlay"):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6,6)); plt.imshow(img)
    plt.scatter(pts[:,0], pts[:,1], c='yellow', s=5)
    plt.title(title); plt.axis('off'); plt.show()

# --------------------------
# 4) Affine warp & triangle morph (unchanged)
# --------------------------
def apply_affine_transform(src, src_tri, dst_tri, size):
    M = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

def morph_triangle(img1, img2, out, t1, t2, t, alpha):
    r1 = cv2.boundingRect(np.float32([t1])); r2 = cv2.boundingRect(np.float32([t2])); r  = cv2.boundingRect(np.float32([t]))
    t1r = [(p[0]-r1[0], p[1]-r1[1]) for p in t1]
    t2r = [(p[0]-r2[0], p[1]-r2[1]) for p in t2]
    tr  = [(p[0]-r[0],  p[1]-r[1])  for p in t ]
    mask = np.zeros((r[3], r[2],3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tr), (1,1,1), 16,0)
    im1 = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    im2 = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    w1 = apply_affine_transform(im1, t1r, tr, (r[2],r[3]))
    w2 = apply_affine_transform(im2, t2r, tr, (r[2],r[3]))
    blended = (1-alpha)*w1 + alpha*w2
    out[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = out[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]*(1-mask) + blended*mask

# --------------------------
# 5) Full morph pipeline
# --------------------------
def morph_head(img1, img2, alpha=0.5):
    # get points
    f1 = get_face_landmarks(img1); f2 = get_face_landmarks(img2)
    c1 = get_head_contour(img1);    c2 = get_head_contour(img2)
    if f1 is None or f2 is None or c1 is None or c2 is None:
        raise RuntimeError("Chýbajú body tváre alebo hlavy.")
    # combine
    pts1 = np.vstack([f1, c1])
    pts2 = np.vstack([f2, c2])
    # interpolated
    pts_mid = ((1-alpha)*pts1 + alpha*pts2).astype(np.int32)

    show_overlay(img1, pts_mid,    "Interpolované body celej hlavy")
    tri = Delaunay(pts_mid)
    # triangulation vizualizácia
    img_tri = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB).copy()
    for t in tri.simplices:
        p = pts_mid[t]
        for i in range(3):
            cv2.line(img_tri, tuple(p[i]), tuple(p[(i+1)%3]), (0,255,0), 1)
    plt.figure(figsize=(6,6)); plt.imshow(img_tri); plt.axis('off'); plt.title("Triangulácia"); plt.show()

    # morph
    out = np.zeros_like(img1, dtype=np.float32)
    for t in tri.simplices:
        morph_triangle(img1, img2, out,
                       [pts1[i] for i in t],
                       [pts2[i] for i in t],
                       [pts_mid[i] for i in t], alpha)
    return cv2.convertScaleAbs(out)

# --------------------------
# 6) Run & show result
# --------------------------
# imgL = cv2.imread("dataset/1/1/0003.png")
imgL = cv2.imread("dataset/dataset/BL.jpeg")
# imgR = cv2.imread("dataset/1/1/0002.png")
imgR = cv2.imread("dataset/dataset/BR.jpeg")
res  = morph_head(imgL, imgR, alpha=0.5)
plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.axis('off'); plt.title("Stredový pohľad hlavy"); plt.show()
