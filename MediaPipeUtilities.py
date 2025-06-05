import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from transformers import pipeline
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from PIL import Image
import torch
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


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
    head_mask[:chin_y+5, :] = 1  # +10px for safety


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

# Load the BiRefNet portrait model
model_id = 'ZhengPeng7/BiRefNet-portrait'  # Portrait-specific model
birefnet = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
birefnet.to(device)
birefnet.eval()

def segment_head(image, output_path=None):
    """
    Segment head from image using BiRefNet
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image with alpha channel
        
    Returns:
        PIL Image with alpha channel (transparent background)
    """
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    if isinstance(image, np.ndarray):
        # Check if image is BGR (OpenCV format) and convert to RGB if needed
        if image.shape[2] == 3:  # If it has 3 channels
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
    else:
        image = image

    input_tensor = transform_image(image).unsqueeze(0).to(device)
    
    # Convert to half precision if using
    if next(birefnet.parameters()).dtype == torch.float16:
        input_tensor = input_tensor.half()
    
    # Run prediction
    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid().cpu()
    
    # Process mask
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    
    # Apply mask to image
    result = image.copy()
    result.putalpha(mask)
    
    # Save result if output path provided
    # if output_path:
    #     result.save(output_path)

    # Convert PIL Image mask to NumPy array
    mask_np = np.array(mask)

    # Ensure mask is normalized between 0 and 1
    if mask_np.max() > 1:
        mask_np = mask_np / 255.0
    
    return mask_np

def compare_generated_to_gt(gen_img, gt_img):
    """
    Compare two images (generated vs ground truth) only in the non-black region of gen_img.
    
    Parameters:
    - gen_img: Generated image (H x W x C), uint8
    - gt_img: Ground truth image (H x W x C), same size and type as gen_img
    
    Returns:
    - metrics: dict with MSE, PSNR, SSIM over the valid region
    """
    # Ensure they have the same shape
    assert gen_img.shape == gt_img.shape, "Generated and GT must match dimensions"

    mask_gen_head = segment_head(gt_img)
    mask_gen_head = (mask_gen_head > 0.5).astype(np.uint8) * 255
    gt_img = cv2.bitwise_and(gt_img, gt_img, mask=mask_gen_head)
    plt.imshow(mask_gen_head)
    plt.show()
    plt.imshow(gt_img)
    plt.show()
    
    # Create mask where generated is not black (any channel non-zero)
    mask = np.any(gen_img != 0, axis=2)
    
    # If no valid pixels, error
    if not mask.any():
        raise ValueError("No non-black pixels found in generated image.")
    
    # Crop to bounding box of mask
    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    gen_crop = gen_img[y1:y2+1, x1:x2+1]
    gt_crop  = gt_img[y1:y2+1, x1:x2+1]
    mask_crop = mask[y1:y2+1, x1:x2+1]

    plt.imshow(gt_crop)
    plt.show()
    plt.imshow(gen_crop)
    plt.show()
    
    # Flatten masked pixels and compute metrics per-channel
    mse_vals, psnr_vals, ssim_vals = [], [], []
    for c in range(gen_crop.shape[2]):
        g = gen_crop[:,:,c][mask_crop]
        t = gt_crop[:,:,c][mask_crop]
        mse_vals.append(mean_squared_error(t, g))
        psnr_vals.append(peak_signal_noise_ratio(t, g, data_range=t.max()-t.min()))
        ssim_vals.append(structural_similarity(t, g, data_range=t.max()-t.min()))
    
    return {
        "MSE": np.mean(mse_vals),
        "PSNR": np.mean(psnr_vals),
        "SSIM": np.mean(ssim_vals)
    }

# ------------------------------------------------------------------
# C) Putting it together
# ------------------------------------------------------------------
if __name__ == "__main__":
    img_path = "dataset/1/1/0001.png"

    img = get_fg(img_path)
    plt.imshow(img)
    plt.show()