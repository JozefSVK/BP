import cv2
import numpy as np

def compute_focus_map_uint8(img, ksize=5):
    """Mapu ostrosti voláme na uint8 obraze."""
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    fm   = np.abs(lap)
    fm   = cv2.GaussianBlur(fm, (ksize, ksize), 0)
    return fm.astype(np.float32) + 1e-6

def region_fusion(warp1, warp2, pts_mid, tri):
    """
    Pre každý trojuholník z tri.simplices:
      - vytvoríme masku trojuholníka
      - spočítame priemernú ostrosť v warp1 vs warp2
      - ak je ostrejší warp1, skopírujeme celý trojuholník z warp1,
        inak z warp2
    """
    H, W = warp1.shape[:2]
    fused = np.zeros_like(warp1, dtype=np.uint8)

    F1 = compute_focus_map_uint8(warp1)
    F2 = compute_focus_map_uint8(warp2)

    for simplex in tri.simplices:
        # body trojuholníka
        tri_pts = pts_mid[simplex]

        # maska trojuholníka
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, tri_pts, 255)

        # získať indexy pixlov v tomto trojuholníku
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue

        # priemerná ostrosť
        f1 = np.mean(F1[ys, xs])
        f2 = np.mean(F2[ys, xs])

        # vybrať warp podľa ostrosti
        src = warp1 if f1 > f2 else warp2

        # skopírovať pixely trojuholníka do výsledku
        fused[ys, xs] = src[ys, xs]

    return fused

def region_fusion_with_mask(w1, w2, pts_mid, tri):
    H,W = w1.shape[:2]
    fused = np.zeros_like(w1, dtype=np.uint8)
    mask  = np.zeros((H,W),      dtype=np.uint8)

    # ostrosť razemosť
    F1 = compute_focus_map_uint8(w1)
    F2 = compute_focus_map_uint8(w2)

    for simplex in tri.simplices:
        tri_pts = pts_mid[simplex]
        # masku trojuholníka
        m = np.zeros((H,W),dtype=np.uint8)
        cv2.fillConvexPoly(m, tri_pts, 255)
        ys,xs = np.nonzero(m)
        if len(xs)==0: continue

        # priemerná ostrosť
        f1 = F1[ys,xs].mean()
        f2 = F2[ys,xs].mean()
        use1 = f1>f2

        # skopírujeme pixely a masku
        if use1:
            fused[ys,xs] = w1[ys,xs]
            mask [ys,xs] = 255
        else:
            fused[ys,xs] = w2[ys,xs]
            # mask zostáva 0
    return fused, mask

def laplacian_pyramid_blend(A, B, mask, levels=4):
    # zmeníme mask na [0..1] float32
    mask = mask.astype(np.float32)/255.0
    # Gaussian pyramids pre mask
    gp_mask = [mask.copy()]
    for i in range(levels):
        gp_mask.append(cv2.pyrDown(gp_mask[-1]))
    # Laplacian pyramids pre A a B
    gpA = [A.astype(np.float32)]
    gpB = [B.astype(np.float32)]
    for i in range(levels):
        gpA.append(cv2.pyrDown(gpA[-1]))
        gpB.append(cv2.pyrDown(gpB[-1]))
    lpA = [gpA[-1]]
    lpB = [gpB[-1]]
    for i in range(levels,0,-1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        la = cv2.subtract(gpA[i-1],
                          cv2.pyrUp(gpA[i], dstsize=size))
        lb = cv2.subtract(gpB[i-1],
                          cv2.pyrUp(gpB[i], dstsize=size))
        lpA.append(la)
        lpB.append(lb)
    # blend pyramids
    LS = []
    for la, lb, gm in zip(lpA, lpB, gp_mask[::-1]):
        gm3 = cv2.merge([gm,gm,gm])
        ls = la*gm3 + lb*(1.0-gm3)
        LS.append(ls)
    # reconstruct
    res = LS[0]
    for i in range(1, levels+1):
        size = (LS[i].shape[1], LS[i].shape[0])
        res = cv2.pyrUp(res, dstsize=size)
        res = cv2.add(res, LS[i])
    return cv2.convertScaleAbs(res)