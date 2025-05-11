import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import demoChat5

# 1) Face Mesh landmarks
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
    pts = [(int(lm.x*w), int(lm.y*h))
           for lm in res.multi_face_landmarks[0].landmark]
    return np.array(pts, dtype=np.int32)

# 2) Head contour pomocou SelfieSegmentation
def get_head_contour(image, n_points=500):
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        res = seg.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    mask = demoChat5.get_fg(image)
    mask = (mask > 0.5).astype(np.uint8) * 255

    # 2) label connected components
    #    convert to 0/1 first so labels are 0..N
    num_labels, labels = cv2.connectedComponents(mask//255)

    # 3) find a seed *inside* the face region
    pts = get_face_landmarks(image)
    if pts is None:
        return None
    # try each landmark until one falls in a non-zero label
    person_label = None
    for (x,y) in pts:
        lab = labels[y, x]
        if lab != 0:
            person_label = lab
            break
    # fallback to centroid if none hit
    if person_label is None:
        cx, cy = np.mean(pts, axis=0).astype(int)
        person_label = labels[cy, cx]

    # 4) build a clean mask of *just* that component
    person_mask = (labels == person_label).astype(np.uint8)*255
    plt.imshow(person_mask)
    plt.show()

    # mask = (res.segmentation_mask > 0.5).astype(np.uint8)*255
    cnts, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    # cnt = max(cnts, key=lambda c: cv2.contourArea(c))[:,0,:]
    # idx = np.round(np.linspace(0, len(cnt)-1, n_points)).astype(int)

    cnt = cnts[0] if len(cnts)==1 else max(cnts, key=cv2.contourArea)
    cnt = cnt[:,0,:]   # reshape

    # 6) down-sample to exactly n_points along the boundary
    L = len(cnt)
    idx = np.round(np.linspace(0, L-1, n_points)).astype(int)
    return cnt[idx]

# 3) Warp cez Delaunay + affine
def warp_images(img1, img2, pts1, pts2, pts_mid):
    tri = Delaunay(pts_mid)
    h, w = img1.shape[:2]
    warp1 = np.zeros((h, w, 3), dtype=np.float32)
    warp2 = np.zeros((h, w, 3), dtype=np.float32)

    def apply_affine(src, dst, tri_src, tri_dst):
        r1 = cv2.boundingRect(np.float32([tri_src]))
        r2 = cv2.boundingRect(np.float32([tri_dst]))
        src_rect = [(p[0]-r1[0], p[1]-r1[1]) for p in tri_src]
        dst_rect = [(p[0]-r2[0], p[1]-r2[1]) for p in tri_dst]
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dst_rect), (1,1,1), 16, 0)
        src_patch = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        M = cv2.getAffineTransform(np.float32(src_rect), np.float32(dst_rect))
        warped = cv2.warpAffine(src_patch, M, (r2[2], r2[3]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = (
            dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]*(1-mask)
            + warped*mask
        )

    for simplex in tri.simplices:
        t1 = [tuple(pts1[i]) for i in simplex]
        t2 = [tuple(pts2[i]) for i in simplex]
        t_mid = [tuple(pts_mid[i]) for i in simplex]
        apply_affine(img1, warp1, t1, t_mid)
        apply_affine(img2, warp2, t2, t_mid)

    return cv2.convertScaleAbs(warp1), cv2.convertScaleAbs(warp2)

# 4) Váhová maska s lineárnym prechodom
def compute_weight_mask(shape, mid, margin, axis="x"):
    H, W = shape[:2]
    if axis == "x":
        coords = np.arange(W, dtype=np.float32)
        w1d = np.clip((coords - (mid - margin)) / (2*margin), 0.0, 1.0)
        mask = np.tile(w1d, (H,1))
    else:  # axis=="y"
        coords = np.arange(H, dtype=np.float32)
        w1d = np.clip((coords - (mid - margin)) / (2*margin), 0.0, 1.0)
        mask = np.tile(w1d[:,None], (1,W))
    return cv2.merge([mask,mask,mask])

# convert contour to a filled polygon mask
def mask_from_contour(img, contour):
    h, w = img.shape[:2]
    m = np.zeros((h,w), dtype=np.uint8)
    # if you want to be extra sure you get the entire forehead/hairline,
    # you can also take a convex hull over pts1 (face+contour).
    # hull = cv2.convexHull(np.vstack([f1, c1]))
    cv2.fillConvexPoly(m, contour, 255)
    return m

# 5) Pipeline vrátane bezpečného seamlessClone
def morph_head(img1, img2, alpha=0.5, margin=50, axis="x"):
    # zber bodov
    f1 = get_face_landmarks(img1); f2 = get_face_landmarks(img2)
    # c1 = get_head_contour(img1);    c2 = get_head_contour(img2)
    c1 = get_head_contour(img1)
    c2 = get_head_contour(img2)
    if f1 is None or f2 is None or c1 is None or c2 is None:
        raise RuntimeError("Chýbajú body.")
    
    plt.imshow(img1)
    plt.scatter(c1[:, 0], c1[:, 1], s=3, c='yellow', alpha=0.5)
    plt.show()
    plt.imshow(img2)
    plt.scatter(c2[:, 0], c2[:, 1], s=3, c='yellow', alpha=0.5)
    plt.show()

    
     # mask out the background using the head contour
    mask1 = mask_from_contour(img1, c1)
    mask2 = mask_from_contour(img2, c2)
    img1_fg = cv2.bitwise_and(img1, img1, mask=mask1)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask2)

    # plt.imshow(img1_fg)
    # plt.show()
    # plt.imshow(img2_fg)
    # plt.show()

    pts1 = np.vstack([f1, c1])
    pts2 = np.vstack([f2, c2])
    pts_mid = ((1-alpha)*pts1 + alpha*pts2).astype(np.int32)

    # warp
    # warp1, warp2 = warp_images(img1, img2, pts1, pts2, pts_mid)
    warp1, warp2 = warp_images(img1_fg, img2_fg, pts1, pts2, pts_mid)

    plt.imshow(warp1)
    plt.show()
    plt.imshow(warp2)
    plt.show()

    # blend pomocou váh
    mid_x = int(np.median(pts_mid[:,0]))
    if axis == "x":
        mid_coord = int(np.median(pts_mid[:,0]))  # legacy left/right
    else:
        mid_coord = int(np.median(pts_mid[:,1]))  # new up/down
    Wmask = compute_weight_mask(img1.shape, mid_coord, margin, axis)
    # plt.imshow(Wmask)
    # plt.show()
    blended = cv2.convertScaleAbs(warp1*(1-Wmask) + warp2*Wmask)

    # získať masku hlavy (pre clone)
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        res = seg.process(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    head = (res.segmentation_mask>0.5).astype(np.uint8)*255

    # boundingRect masky (bezpečný center)
    x,y,w,h = cv2.boundingRect(head)
    center = (x + w//2, y + h//2)

    # seamlessClone (iba raz, už to nespadne mimo)
    result = cv2.seamlessClone(blended, blended, head, center, cv2.NORMAL_CLONE)
    return result

# 6) Spustenie
if __name__ == "__main__":
    # imgL = cv2.imread("dataset/1/1/0003.png")
    # imgR = cv2.imread("dataset/1/1/0002.png")
    # # imgL = cv2.imread("dataset/dataset/BL.jpeg")
    # # imgR = cv2.imread("dataset/dataset/BR.jpeg")
    # res = morph_head(imgL, imgR, alpha=0.5, margin=60)
    # # cv2.imwrite("output/morphed_fixed.png", res)
    # # fg, bg_mask = demoChat5.remove_background(res, thresh=0.5)
    # # plt.imshow(fg)
    # # plt.show()
    # res = demoChat5.extract_head_only(res)
    # # res = demoChat5.crop_head_rgba(res)
    # plt.figure(figsize=(6,6))
    # plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    # plt.axis('off'); plt.title("Morph – opravené seamlessClone"); plt.show()


    # imgL = cv2.imread("dataset/1/1/0005.png")
    # imgR = cv2.imread("dataset/1/1/0004.png")
    imgL = cv2.imread("dataset/dataset/TL.jpeg")
    imgR = cv2.imread("dataset/dataset/BL.jpeg")
    res = morph_head(imgL, imgR, alpha=0.5, margin=60)
    # cv2.imwrite("output/morphed_fixed.png", res)
    # fg, bg_mask = demoChat5.remove_background(res, thresh=0.5)
    # plt.imshow(fg)
    # plt.show()
    res = demoChat5.extract_head_only(res)
    # res = demoChat5.crop_head_rgba(res)
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.axis('off'); plt.title("Morph – opravené seamlessClone"); plt.show()
