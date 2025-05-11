import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
import torch


# ------------------------------------------------------------------
# A) Remove *all* background (only keep pixels the SelfieSegmentation
#    thinks are “you”). This happens *before* we even look at the chin.
# ------------------------------------------------------------------
def remove_background(image, thresh=0.5):
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        res = seg.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # binary mask of “person” vs background
    mask = (res.segmentation_mask > thresh).astype(np.uint8)
    # apply it
    fg = cv2.bitwise_and(image, image, mask=mask)
    return fg, mask

# ------------------------------------------------------------------
# B) Then extract just the head (above the chin), exactly as before
# ------------------------------------------------------------------
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
    return np.array([
        (int(lm.x*w), int(lm.y*h))
        for lm in res.multi_face_landmarks[0].landmark
    ], dtype=np.int32)

def extract_head_only(fg_image):
    h, w = fg_image.shape[:2]
    # 1) chin landmark
    pts = get_face_landmarks(fg_image)
    if pts is None:
        raise RuntimeError("FaceMesh nenašlo žiadnu tvár.")
    chin_y = pts[152][1]

    # 2) build a mask that is “1” only above the chin
    head_mask = np.zeros((h, w), dtype=np.uint8)
    head_mask[:chin_y+10, :] = 1  # +10px for safety


    # 3) optional: smooth the top edge so it isn’t jaggy
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(head_mask)
    # plt.show()
    # 4) apply to the already background-removed image
    b,g,r = cv2.split(fg_image)
    alpha = (head_mask * 255).astype(np.uint8)
    res = cv2.bitwise_and(fg_image, fg_image, mask=head_mask)
    # return cv2.merge((b,g,r,alpha))
    return res

def crop_head_rgba(rgba):
    """
    Given an RGBA image where A=255 on the head and A=0 elsewhere,
    crop to the minimal rectangle containing all non-zero alpha pixels.
    """
    # extract alpha channel
    alpha = rgba[:,:,3]
    # find bounding box of non-zero alpha
    ys, xs = np.where(alpha > 0)
    if len(xs)==0 or len(ys)==0:
        # nothing to crop
        return rgba
    x, y, w, h = cv2.boundingRect(np.stack([xs, ys], axis=1))
    # crop all 4 channels
    return rgba[y:y+h, x:x+w, :]

pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
def get_fg(img_path):
    # Convert the NumPy array to PIL Image
    img_pil = Image.fromarray(img_path)
    
    # Get the mask as PIL Image
    pillow_mask = pipe(img_pil, return_mask=True)
    
    # Convert PIL Image mask to NumPy array
    mask_np = np.array(pillow_mask)

    # Ensure mask is normalized between 0 and 1
    if mask_np.max() > 1:
        mask_np = mask_np / 255.0

    return mask_np


# ------------------------------------------------------------------
# C) Putting it together
# ------------------------------------------------------------------
if __name__ == "__main__":
    img_path = "dataset/1/1/0001.png"

    img = get_fg(img_path)
    plt.imshow(img)
    plt.show()