import cv2
import numpy as np
import maxflow

def graph_cut_seam(w1, w2, alpha):
    """
    Nájde binárnu masku (0/1) podľa ktorého zabudujeme warp1 vs warp2.
    """
    h, w = w1.shape[:2]
    # 1) Energia na pixely: gradient magnitude vo výsledku
    gray1 = cv2.cvtColor(w1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(w2, cv2.COLOR_BGR2GRAY)
    # Silueta: vyšší rozdiel = väčší “cost” pri prechode
    grad1 = np.abs(cv2.Laplacian(gray1, cv2.CV_64F))
    grad2 = np.abs(cv2.Laplacian(gray2, cv2.CV_64F))
    # Penalty pri prechode medzi warp1→warp2 = priemer gradientov
    cost = (grad1 + grad2) / 2.0
    cost = (cost / cost.max() * 100.0).astype(np.int32) + 1  # kladne váhy

    # 2) Graph-cut definícia
    g = maxflow.Graph[int](h*w, 4*h*w)
    nodes = g.add_nodes(h*w)

    # Pre každý pixel pridajeme hrany do S (warp1) a T (warp2)
    # váhy podľa toho, ako veľmi chceme daný pixel z warp1 vs warp2
    # tu pre jednoduchosť považujeme cost ako symetrický
    for y in range(h):
        for x in range(w):
            idx = y*w + x

            C = 100
            bias_L = (1- alpha) * C
            bias_R = alpha * C

            # dataterm: nízky cost → bližšie k warp1, vysoký → k warp2
            g.add_tedge(idx, 100 - cost[y,x], cost[y,x])
            # smoothness term medzi susedmi 4-smerne
            if x+1 < w:
                w_p = cost[y,x] + cost[y,x+1]
                g.add_edge(idx, idx+1, w_p, w_p)
            if y+1 < h:
                w_p = cost[y,x] + cost[y+1,x]
                g.add_edge(idx, idx+w, w_p, w_p)

    g.maxflow()
    mask = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            idx = y*w + x
            mask[y, x] = 255 if g.get_segment(idx) == 0 else 0
    return mask
