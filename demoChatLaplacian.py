import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_focus_map(img, ksize=5):
    """Ostrosť = variance of Laplacian, potom Gaussian pre hladšie výsledky."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    fm  = cv2.convertScaleAbs(lap)              # absolútna hodnota
    fm  = cv2.GaussianBlur(fm, (ksize,ksize), 0)  # trochu rozmazať
    return fm.astype(np.float32) + 1e-6          # +eps, aby sme nedelili nulou

def focus_fusion(warp1, warp2):
    """Spojí warp1 a warp2 podľa lokálnej ostrosti."""
    F1 = compute_focus_map(warp1)
    F2 = compute_focus_map(warp2)
    W1 = F1 / (F1 + F2)
    W2 = 1.0 - W1

    # Vizualizácia váh (len pre kontrolu)
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1); plt.title("W1 – váhy pre warp1"); plt.imshow(W1, cmap='viridis'); plt.axis('off')
    plt.subplot(1,2,2); plt.title("W2 – váhy pre warp2"); plt.imshow(W2, cmap='viridis'); plt.axis('off')
    plt.show()

    # Pre každý kanál BGR vykonáme váhovaný blend
    fused = (warp1.astype(np.float32) * W1[:,:,None] +
             warp2.astype(np.float32) * W2[:,:,None])
    return cv2.convertScaleAbs(fused)


# --- Predpokladáme, že warp1 a warp2 už máme z morph_head ---
# warp1 = ... 
# warp2 = ...

# fused = focus_fusion(warp1, warp2)
# plt.figure(figsize=(6,6))
# plt.imshow(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB))
# plt.title("Fúzia podľa ostrosti"); plt.axis('off'); plt.show()
