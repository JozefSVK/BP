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
            min_detection_confidence=0.4) as fm:
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

    mask = demoChat5.segment_head(image)
    mask = (mask > 0.5).astype(np.uint8) * 255
    # plt.imshow(mask)
    # plt.show()

    # mask = (res.segmentation_mask > 0.5).astype(np.uint8)*255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=lambda c: cv2.contourArea(c))[:,0,:]
    idx = np.round(np.linspace(0, len(cnt)-1, n_points)).astype(int)

    # cnt = cnts[0] if len(cnts)==1 else max(cnts, key=cv2.contourArea)
    # cnt = cnt[:,0,:]   # reshape

    # # 6) down-sample to exactly n_points along the boundary
    # L = len(cnt)
    # idx = np.round(np.linspace(0, L-1, n_points)).astype(int)
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

def synthesize_mid_from_up_down(warp_top, warp_bot, pts_mid, hairline_margin=20, chin_margin=20):
    H, W = warp_top.shape[:2]
    # 1) find hairline and chin y from your mid‐points
    all_y = pts_mid[:,1]
    hairline_y = int(all_y.min())            # highest point of your mid‐mesh
    chin_y     = int(all_y.max())            # lowest point

    # 2) define three bands:
    #    • top band: y < hairline_y + hairline_margin  → copy warp_top
    #    • bottom band: y > chin_y - chin_margin       → copy warp_bot
    #    • middle band: blend
    Y = np.arange(H)[:,None]
    Y2 = np.tile(Y, (1, W))                  # shape (H,W)
    
    # 3) three bands
    top_mask    = (Y2 <  hairline_y + hairline_margin)
    bottom_mask = (Y2 >  chin_y    - chin_margin)
    middle_mask = ~(top_mask | bottom_mask)

    # 4) build a linear ramp only in the middle band
    start = hairline_y + hairline_margin
    end   = chin_y    - chin_margin
    ramp = np.zeros_like(Y2, dtype=np.float32)
    if end > start:
        ramp_vals = (Y2 - start) / float(end - start)
        ramp = np.clip(ramp_vals, 0.0, 1.0)
    # expand to 3‐channels
    Wmask = np.stack([ramp]*3, axis=2)       # shape (H,W,3)

    # 5) assemble final
    out = np.zeros_like(warp_top)
    # copy pure top region
    out[top_mask] = warp_top[top_mask]
    # copy pure bottom region
    out[bottom_mask] = warp_bot[bottom_mask]
    # blend only in the middle
    blended = cv2.convertScaleAbs(warp_top*(1-Wmask) + warp_bot*Wmask)
    out[middle_mask] = blended[middle_mask]

    return out

def densify_points(pts, max_edge_len=20, n_samples=1):
    """
    Given an (N,2) array of points, build a Delaunay triangulation,
    look at every edge: if its length > max_edge_len, insert `n_samples`
    equally spaced points along that edge. Repeat until no edges too long.
    Returns a new (M,2) array of points.
    """
    pts = pts.copy()
    while True:
        tri = Delaunay(pts)
        # collect all unique edges
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = sorted([simplex[i], simplex[(i+1)%3]])
                edges.add((a,b))
        edges = list(edges)
        
        # find which edges are too long
        to_add = []
        for i,j in edges:
            p, q = pts[i], pts[j]
            dist = np.linalg.norm(p-q)
            if dist > max_edge_len:
                # sample points between p and q
                for k in range(1, n_samples+1):
                    t = k/(n_samples+1)
                    to_add.append(p*(1-t) + q*t)
        
        if not to_add:
            break  # all edges short enough
        
        # append new points and loop again
        pts = np.vstack([pts, np.array(to_add)])
    return pts.astype(np.int32)

# 5) Pipeline vrátane bezpečného seamlessClone
def morph_head(img1, img2, alpha=0.5, margin=50, axis="x"):
    # zber bodov
    f1 = get_face_landmarks(img1); f2 = get_face_landmarks(img2)
    # c1 = get_head_contour(img1);    c2 = get_head_contour(img2)
    c1 = get_head_contour(img1)
    c2 = get_head_contour(img2)
    print(f1 is None)
    print(f2 is None)
    print(c1 is None)
    print(c2 is None)
    if f1 is None or f2 is None or c1 is None or c2 is None:
        raise RuntimeError("Chýbajú body.")
    
    save_points_to_txt_structured("vstupne_body_img1.txt",
                                face_landmarks_f1=f1,
                                head_contour_c1=c1)
    
    # Uloženie vstupných bodov pre img2
    save_points_to_txt_structured("vstupne_body_img2.txt",
                                  face_landmarks_f2=f2,
                                  head_contour_c2=c2)
    
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    plt.imshow(img1_rgb)
    plt.scatter(f1[:, 0], f1[:, 1], s=3, c='yellow', alpha=0.5)
    plt.show()
    plt.imshow(img2_rgb)
    plt.scatter(f2[:, 0], f2[:, 1], s=3, c='yellow', alpha=0.5)
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

        # PRED densify_points (ak by si pts_mid tiež densifikoval, tak by si ukladal tie)
    num_face_landmarks_f1 = f1.shape[0]
    pts_mid_face_component = pts_mid[:num_face_landmarks_f1]
    pts_mid_contour_component = pts_mid[num_face_landmarks_f1:]

    save_points_to_txt_structured("stredove_body_pts_mid_komponenty.txt",
                                  face_component=pts_mid_face_component,
                                  contour_component=pts_mid_contour_component)

    plt.imshow(img1_rgb)
    plt.scatter(pts1[:, 0], pts1[:, 1], s=3, c='yellow', alpha=0.5)
    plt.show()
    plt.imshow(img2_rgb)
    plt.scatter(pts2[:, 0], pts2[:, 1], s=3, c='yellow', alpha=0.5)
    plt.show()

    pts1 = densify_points(pts1, 70, 1)
    head1 = demoChat5.extract_head_only(img1_fg)
    plt.imshow(head1)
    plt.scatter(pts1[:, 0], pts1[:, 1], s=3, c='yellow', alpha=0.5)
    plt.show()
    head2 = demoChat5.extract_head_only(img2_fg)
    plt.imshow(head2)
    plt.scatter(pts2[:, 0], pts2[:, 1], s=3, c='yellow', alpha=0.5)
    plt.show()

    # warp
    # warp1, warp2 = warp_images(img1, img2, pts1, pts2, pts_mid)
    warp1, warp2 = warp_images(img1_fg, img2_fg, pts1, pts2, pts_mid)

    plt.imshow(cv2.cvtColor(warp1, cv2.COLOR_BGR2RGB))
    plt.show()
    plt.imshow(cv2.cvtColor(warp2, cv2.COLOR_BGR2RGB))
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

    final = synthesize_mid_from_up_down(
        warp1, warp2,
        pts_mid,
        hairline_margin=20,
        chin_margin=20
    )

    # plt.imshow(final)
    # plt.title("zmena")
    # plt.show()

    # získať masku hlavy (pre clone)
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        res = seg.process(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    head = (res.segmentation_mask>0.5).astype(np.uint8)*255

    # boundingRect masky (bezpečný center)
    x,y,w,h = cv2.boundingRect(head)
    center = (x + w//2, y + h//2)

    # seamlessClone (iba raz, už to nespadne mimo)
    result = cv2.seamlessClone(blended, blended, head, center, cv2.NORMAL_CLONE)
    
    chin_y = pts_mid[152][1]

    # 2) build a mask that is “1” only above the chin
    h, w = img1.shape[:2]
    head_mask = np.zeros((h, w), dtype=np.uint8)
    head_mask[:chin_y+5, :] = 1  # +10px for safety


    # 3) optional: smooth the top edge so it isn’t jaggy
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, kernel)

    res = cv2.bitwise_and(result, result, mask=head_mask)
    # plt.imshow(res)
    # plt.show()
    return res

def save_points_to_txt_structured(filename, **kwargs):
    """
    Uloží viacero sád bodov do jedného textového súboru so sekciami.
    Každá sada bodov je pomenovaná kľúčom z kwargs.

    Args:
        filename (str): Názov súboru na uloženie.
        **kwargs: Kľúčové argumenty, kde kľúč je názov sekcie (napr. "face_landmarks")
                  a hodnota je NumPy pole bodov (N, 2) alebo None.
    """
    with open(filename, 'w') as f:
        for name, points_array in kwargs.items():
            f.write(f"--- {name} ---\n")
            if points_array is None:
                f.write("None\n")
            else:
                for point in points_array:
                    f.write(f"{int(point[0])} {int(point[1])}\n") # Ukladáme ako celé čísla
            f.write("\n") # Prázdny riadok medzi sekciami pre lepšiu čitateľnosť
    print(f"Štruktúrované body uložené do {filename}")

# 6) Spustenie
if __name__ == "__main__":
    imgL = cv2.imread("dataset/1/1/0003.png")
    imgR = cv2.imread("dataset/1/1/0002.png")
    # imgL = cv2.imread("new_res_head/L5.png")
    # imgL = cv2.resize(imgL, None, fx=2, fy=2)
    # imgR = cv2.imread("new_res_head/R5.png")
    # imgR = cv2.resize(imgR, None, fx=2, fy=2)
    # imgL = cv2.imread("dataset/dataset/TL.jpeg")
    # imgR = cv2.imread("dataset/dataset/TR.jpeg")
    # imgL = cv2.rotate(imgL, cv2.ROTATE_90_CLOCKWISE)
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
    # imgL = cv2.imread("dataset/dataset/TL.jpeg")
    # imgR = cv2.imread("dataset/dataset/BL.jpeg")
    # imgL = cv2.resize(imgL, None, fx=1/3, fy=1/3)
    # imgR = cv2.resize(imgR, None, fx=1/3, fy=1/3)
    res = morph_head(imgL, imgR, alpha=0.5, margin=60, axis="x")
    # cv2.imwrite("output/morphed_fixed.png", res)
    # fg, bg_mask = demoChat5.remove_background(res, thresh=0.5)
    # plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    # plt.show()

    gt = cv2.imread("dataset/1/1/0001.png")
    metrics = demoChat5.compare_generated_to_gt(res, gt)
    print(metrics)
    # res = demoChat5.crop_head_rgba(res)
    plt.figure(figsize=(6,6))
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    plt.axis('off'); plt.title("Morph – opravené seamlessClone"); plt.show()
