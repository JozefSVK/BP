import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import demoChatLaplacian
import demoChatRegionbased
import demoChatGraphCut

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
    return np.array(pts, dtype=np.int32)  # (478,2)

# 2) Head silhouette via SelfieSegmentation
def get_head_contour(image, n_points=500):
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        res = seg.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask = (res.segmentation_mask > 0.5).astype(np.uint8)*255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=lambda c: cv2.contourArea(c))[:,0,:]
    L = len(cnt)
    idx = np.round(np.linspace(0, L-1, n_points)).astype(int)
    return cnt[idx]  # (n_points,2)

# 3) Affine warp helper
def apply_affine_transform(src, src_tri, dst_tri, size):
    M = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    return cv2.warpAffine(src, M, size,
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)

# 4) Triangle morph into a target canvas
def morph_triangle_into(img_src, canvas, t_src, t_mid):
    # bounding rects
    r_src = cv2.boundingRect(np.float32([t_src]))
    r_mid = cv2.boundingRect(np.float32([t_mid]))
    # offset triangles
    src_rect = [(p[0]-r_src[0], p[1]-r_src[1]) for p in t_src]
    mid_rect = [(p[0]-r_mid[0], p[1]-r_mid[1]) for p in t_mid]
    # mask
    mask = np.zeros((r_mid[3], r_mid[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(mid_rect), (1,1,1), 16,0)
    # crops
    crop = img_src[r_src[1]:r_src[1]+r_src[3],
                   r_src[0]:r_src[0]+r_src[2]]
    # warp
    warped = apply_affine_transform(crop, src_rect, mid_rect,
                                    (r_mid[2], r_mid[3]))
    # blend
    canvas[r_mid[1]:r_mid[1]+r_mid[3],
           r_mid[0]:r_mid[0]+r_mid[2]] = (
        canvas[r_mid[1]:r_mid[1]+r_mid[3],
               r_mid[0]:r_mid[0]+r_mid[2]] * (1-mask)
        + warped * mask
    )

# 5) Upravený pipline
def morph_head(img1, img2, alpha=0.5):
    # body
    f1 = get_face_landmarks(img1); f2 = get_face_landmarks(img2)
    c1 = get_head_contour(img1);    c2 = get_head_contour(img2)
    if f1 is None or f2 is None or c1 is None or c2 is None:
        raise RuntimeError("Chýbajú body.")
    pts1 = np.vstack([f1, c1])
    pts2 = np.vstack([f2, c2])
    pts_mid = ((1-alpha)*pts1 + alpha*pts2).astype(np.int32)

    # triangulácia
    tri = Delaunay(pts_mid)

    # canvasy pre warp z ľava a z prava
    warp1 = np.zeros_like(img1, dtype=np.float32)
    warp2 = np.zeros_like(img2, dtype=np.float32)

    # morphing každého trojuholníka
    for simplex in tri.simplices:
        t1 = [pts1[i] for i in simplex]
        t2 = [pts2[i] for i in simplex]
        t_mid = [pts_mid[i] for i in simplex]
        morph_triangle_into(img1, warp1, t1, t_mid)
        morph_triangle_into(img2, warp2, t2, t_mid)

    # zobrazíme najprv obidva warpy
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(warp1),
                            cv2.COLOR_BGR2RGB))
    plt.title("Warp ľavého obrazu na stredové body"); plt.axis('off'); plt.show()

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(cv2.convertScaleAbs(warp2),
                            cv2.COLOR_BGR2RGB))
    plt.title("Warp pravého obrazu na stredové body"); plt.axis('off'); plt.show()

    # finálny blend
    warp1_u8 = cv2.convertScaleAbs(warp1)
    warp2_u8 = cv2.convertScaleAbs(warp2)
    xs = pts_mid[:,0]
    mid_x = int(np.median(xs))   # alebo mid_x = img.shape[1]//2
    H, W = warp1.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    # ľavú polovicu označíme 255, pravú 0
    mask[:, :mid_x] = 255

    # final = ((1-alpha)*warp1 + alpha*warp2)
    # final = np.where(mask[:,:,None]==255, warp1_u8, warp2_u8)
    final = demoChatLaplacian.focus_fusion(warp1_u8, warp2_u8)
    # final = demoChatRegionbased.region_fusion(cv2.convertScaleAbs(warp1), cv2.convertScaleAbs(warp2), pts_mid, tri)
    # final, mask = demoChatRegionbased.region_fusion_with_mask(cv2.convertScaleAbs(warp1), cv2.convertScaleAbs(warp2), pts_mid, tri)
    # mask_gc = demoChatGraphCut.graph_cut_seam(warp1_u8, warp2_u8, alpha)

    # fused_gc = np.where(mask_gc[..., None]==255, warp1_u8, warp2_u8)
    # # vizualizácia masky a predbežného výsledku
    # plt.figure(figsize=(12,4))
    # plt.subplot(1,3,1)
    # plt.title("Maska Graph-cut")
    # plt.imshow(mask_gc, cmap='gray'); plt.axis('off')
    # plt.subplot(1,3,2)
    # plt.title("Warp1 podľa masky")
    # plt.imshow(cv2.cvtColor(fused_gc, cv2.COLOR_BGR2RGB))
    # plt.axis('off')

    # # nájdeme stredečok masky pre seamlessClone
    # ys, xs = np.nonzero(mask_gc)
    # center = (int(xs.mean()), int(ys.mean()))

    # # cv2.NORMAL_CLONE alebo MIXED_CLONE
    # poisson = cv2.seamlessClone(
    #     fused_gc,        # src (fused)
    #     fused_gc,        # dst (rovnaký, len chceme vyhladiť hranice)
    #     mask_gc,         # mask
    #     center,
    #     cv2.NORMAL_CLONE
    # )

    # plt.subplot(1,3,3)
    # plt.title("Poisson (seamless) blending")
    # plt.imshow(cv2.cvtColor(poisson, cv2.COLOR_BGR2RGB)); plt.axis('off')
    # plt.show()

    final = cv2.convertScaleAbs(final)

    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title("Finálny stredový morph hlavy"); plt.axis('off'); plt.show()

    # plt.imshow(mask)
    # plt.show()

    # final = demoChatRegionbased.laplacian_pyramid_blend(cv2.convertScaleAbs(warp1), cv2.convertScaleAbs(warp2), mask, levels=5)

    # plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    # plt.title("Finálny stredový morph hlavy"); plt.axis('off'); plt.show()

    return final

# 6) Spustenie
imgL = cv2.imread("dataset/2/0003.png")
imgR = cv2.imread("dataset/2/0002.png")
# imgL = cv2.imread("dataset/dataset/BL.jpeg")
# imgR = cv2.imread("dataset/dataset/BR.jpeg")
morphed = morph_head(imgL, imgR, alpha=0.2)
