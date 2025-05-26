import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import Delaunay
import demoChat5
import matplotlib.pyplot as plt

def get_face_landmarks(image, refine_landmarks=True, min_confidence=0.4):
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_confidence
    ) as mesh:
        res = mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None
    h, w = image.shape[:2]
    pts = [(int(lm.x*w), int(lm.y*h))
           for lm in res.multi_face_landmarks[0].landmark]
    return np.array(pts, dtype=np.int32)

def get_head_contour(image, face_pts, n_points=2000):
    # demoChat5.segment_head vráti masku [0..1]
    mask = (demoChat5.segment_head(image) > 0.5).astype(np.uint8)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=lambda c: cv2.contourArea(c))[:,0,:]
    all_pts = np.vstack([cnt, face_pts])
    hull    = cv2.convexHull(all_pts)[:,0,:]
    idx     = np.round(np.linspace(0, len(hull)-1, n_points)).astype(int)
    return hull[idx]

def densify_triplet(pts1, pts2, pts_mid, max_edge=20):
    pts1 = pts1.astype(np.float32)
    pts2 = pts2.astype(np.float32)
    ptsm = pts_mid.astype(np.float32)
    while True:
        tri = Delaunay(ptsm)
        edges = set()
        for s in tri.simplices:
            for i in range(3):
                a,b = sorted([s[i], s[(i+1)%3]])
                edges.add((a,b))
        to_add = []
        for i,j in edges:
            if np.linalg.norm(ptsm[i]-ptsm[j]) > max_edge:
                t = 0.5
                to_add.append((
                    pts1[i]*(1-t) + pts1[j]*t,
                    pts2[i]*(1-t) + pts2[j]*t,
                    ptsm[i]*(1-t) + ptsm[j]*t
                ))
        if not to_add:
            break
        for n1,n2,nm in to_add:
            pts1  = np.vstack([pts1,  n1])
            pts2  = np.vstack([pts2,  n2])
            ptsm = np.vstack([ptsm, nm])
    return pts1.astype(np.int32), pts2.astype(np.int32), ptsm.astype(np.int32)

def warp_images(img1, img2, pts1, pts2, pts_mid):
    tri = Delaunay(pts_mid)
    h,w = img1.shape[:2]
    w1 = np.zeros((h,w,3), np.float32)
    w2 = np.zeros((h,w,3), np.float32)
    def apply_affine(src, dst, src_tri, dst_tri):
        r1 = cv2.boundingRect(np.float32([src_tri]))
        r2 = cv2.boundingRect(np.float32([dst_tri]))
        src_rect = [(p[0]-r1[0], p[1]-r1[1]) for p in src_tri]
        dst_rect = [(p[0]-r2[0], p[1]-r2[1]) for p in dst_tri]
        mask = np.zeros((r2[3],r2[2],3), np.float32)
        cv2.fillConvexPoly(mask, np.int32(dst_rect), (1,1,1), 16,0)
        patch = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        M = cv2.getAffineTransform(np.float32(src_rect), np.float32(dst_rect))
        warped = cv2.warpAffine(patch, M, (r2[2],r2[3]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = (
            dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]*(1-mask)
            + warped*mask
        )
    for simplex in tri.simplices:
        t1 = [tuple(pts1[i]) for i in simplex]
        t2 = [tuple(pts2[i]) for i in simplex]
        tm = [tuple(pts_mid[i]) for i in simplex]
        apply_affine(img1, w1, t1, tm)
        apply_affine(img2, w2, t2, tm)
    return cv2.convertScaleAbs(w1), cv2.convertScaleAbs(w2)

def compute_weight_mask(shape, mid, margin, axis="x"):
    H,W = shape[:2]
    if axis=="x":
        coords = np.arange(W, dtype=np.float32)
        w1d = np.clip((coords - (mid-margin))/(2*margin),0,1)
        mask = np.tile(w1d, (H,1))
    else:
        coords = np.arange(H, dtype=np.float32)
        w1d = np.clip((coords - (mid-margin))/(2*margin),0,1)
        mask = np.tile(w1d[:,None], (1,W))
    return cv2.merge([mask,mask,mask])

def synthesize_mid(w1, w2, pts_mid, hairline_margin=20, chin_margin=20):
    H,W = w1.shape[:2]
    ys  = pts_mid[:,1]
    hair, chin = int(ys.min()), int(ys.max())
    Y = np.arange(H)[:,None].repeat(W,axis=1)
    top = (Y < hair+hairline_margin)
    bot = (Y > chin-chin_margin)
    mid = ~(top|bot)
    ramp = np.zeros_like(Y, np.float32)
    start, end = hair+hairline_margin, chin-chin_margin
    if end>start:
        ramp = np.clip((Y-start)/(end-start),0,1)
    Wmask = np.stack([ramp]*3,axis=2)
    out   = np.zeros_like(w1)
    out[top] = w1[top]
    out[bot] = w2[bot]
    blend   = cv2.convertScaleAbs(w1*(1-Wmask)+w2*Wmask)
    out[mid] = blend[mid]
    return out

def mask_from_contour(img, contour):
    h,w = img.shape[:2]
    m = np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(m, contour, 255)
    return m

def morph_head(img1, img2, alpha=0.5, margin=50, axis="x"):
    f1 = get_face_landmarks(img1)
    f2 = get_face_landmarks(img2)
    if f1 is None or f2 is None:
        raise RuntimeError("FaceMesh zlyhal.")
    c1 = get_head_contour(img1, f1)
    c2 = get_head_contour(img2, f2)
    if c1 is None or c2 is None:
        raise RuntimeError("Head contour zlyhal.")

    # foreground
    m1, m2 = mask_from_contour(img1,c1), mask_from_contour(img2,c2)
    fg1, fg2 = cv2.bitwise_and(img1,img1,mask=m1), cv2.bitwise_and(img2,img2,mask=m2)

    # body-points + mid
    pts1    = np.vstack([f1,c1])
    pts2    = np.vstack([f2,c2])
    pts_mid = ((1-alpha)*pts1 + alpha*pts2).astype(np.int32)

    # densify all three at once
    pts1, pts2, pts_mid = densify_triplet(pts1,pts2,pts_mid, max_edge=20)

    # warp both into mid
    w1, w2 = warp_images(fg1, fg2, pts1, pts2, pts_mid)

    # simple blend (voliteľné)
    mid_x = int(np.median(pts_mid[:,0])) if axis=="x" else int(np.median(pts_mid[:,1]))
    Wm    = compute_weight_mask(img1.shape, mid_x, margin, axis)
    blended = cv2.convertScaleAbs(w1*(1-Wm)+w2*Wm)

    # horná/spodná synthéza
    final_raw = synthesize_mid(w1,w2, pts_mid)

    # orež podľa mid kontúry
    cnt_mid     = pts_mid[len(f1):]
    mask_mid    = mask_from_contour(final_raw, cnt_mid)
    # --- vyhladenie masky ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35))
    mask_mid = cv2.morphologyEx(mask_mid, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Gaussian blur + normalizácia na [0,1]
    mask_smooth = cv2.GaussianBlur(mask_mid.astype(np.float32), (31,31), 0) / 255.0
    # z jednotlivých kanálov vyrobíme 3-kanálovú váhu
    mask_smooth = np.stack([mask_smooth]*3, axis=2)
    # aplikuj na final_raw
    final_crop  = (final_raw.astype(np.float32) * mask_smooth).astype(np.uint8)

    # posuň do stredu výstupného plátna
    H,W   = final_crop.shape[:2]
    canvas = np.zeros_like(final_crop)
    cnt_mid   = pts_mid[len(f1):]  # body kontúry
    # 1) prahové hodnoty
    left_thresh  = int(np.percentile(cnt_mid[:,0],  5))
    right_thresh = int(np.percentile(cnt_mid[:,0], 95))
    # 2) masky
    left_mask   = np.zeros((H,W), bool)
    right_mask  = np.zeros((H,W), bool)
    left_mask[:, :left_thresh]    = True
    right_mask[:, right_thresh:]  = True
    center_mask = ~(left_mask | right_mask)
    # 3) zloženie uší aj stredu
    canvas = np.zeros_like(final_crop)
    canvas[left_mask]   = w1[left_mask]
    canvas[right_mask]  = w2[right_mask]
    canvas[center_mask] = final_crop[center_mask]
    return canvas
    # offset aby stred heddu sadol do stredu obrazka
    cur_mid = int(np.median(cnt_mid[:,0]))
    desired = W//2
    dx = desired - cur_mid
    M  = np.float32([[1,0,dx],[0,1,0]])
    # canvas = cv2.warpAffine(final_crop, M, (W,H),
    #                         flags=cv2.INTER_LINEAR,
    #                         borderMode=cv2.BORDER_CONSTANT,
    #                         borderValue=(0,0,0))
    canvas = cv2.warpAffine(final_crop, M, (W,H),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)

    return canvas

if __name__ == "__main__":
    imgL = cv2.imread("dataset/1/1/0003.png")
    imgR = cv2.imread("dataset/1/1/0002.png")
    result = morph_head(imgL, imgR, alpha=0.5, margin=60, axis="x")

    # zobraz cez matplotlib
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Morphed Head – centrálny pohľad")
    plt.show()
