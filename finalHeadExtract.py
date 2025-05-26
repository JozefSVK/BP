import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from transformers import pipeline, AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image
import torch
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


class HeadSegmentor:
    """Centralized head segmentation with multiple backends"""
    
    def __init__(self, method='mediapipe'):
        self.method = method
        self._init_models()
    
    def _init_models(self):
        if self.method == 'mediapipe':
            self.seg_model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5
            )
        elif self.method == 'birefnet':
            model_id = 'ZhengPeng7/BiRefNet-portrait'
            self.birefnet = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.birefnet.to(self.device)
            self.birefnet.eval()
            self.transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif self.method == 'rmbg':
            self.pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    
    def get_head_mask(self, image):
        """Get head segmentation mask"""
        if self.method == 'mediapipe':
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = self.seg_model.process(rgb_image)
            return (res.segmentation_mask > 0.5).astype(np.uint8) * 255
        
        elif self.method == 'birefnet':
            return self._segment_with_birefnet(image)
        
        elif self.method == 'rmbg':
            return self._segment_with_rmbg(image)
    
    def _segment_with_birefnet(self, image):
        """BiRefNet segmentation implementation"""
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        if next(self.birefnet.parameters()).dtype == torch.float16:
            input_tensor = input_tensor.half()
        
        with torch.no_grad():
            preds = self.birefnet(input_tensor)[-1].sigmoid().cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        mask_np = np.array(mask)
        
        if mask_np.max() > 1:
            mask_np = mask_np / 255.0
        
        return (mask_np * 255).astype(np.uint8)
    
    def _segment_with_rmbg(self, image):
        """RMBG segmentation implementation"""
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pillow_mask = self.pipe(img_pil, return_mask=True)
        mask_np = np.array(pillow_mask)
        
        if mask_np.max() > 1:
            mask_np = mask_np / 255.0
        
        return (mask_np * 255).astype(np.uint8)
    
    def get_face_landmarks(self, image):
        """Extract face landmarks"""
        if self.method != 'mediapipe':
            # Create temporary face mesh for landmark extraction
            with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5) as fm:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb_image)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb_image)
        
        if not res.multi_face_landmarks:
            return None
        
        h, w = image.shape[:2]
        return np.array([
            (int(lm.x*w), int(lm.y*h))
            for lm in res.multi_face_landmarks[0].landmark
        ], dtype=np.int32)
    
    def extract_head_only(self, image):
        """Extract head region (above chin)"""
        # Get landmarks to find chin
        landmarks = self.get_face_landmarks(image)
        if landmarks is None:
            raise RuntimeError("Could not detect face landmarks")
        
        chin_y = landmarks[152][1]  # Chin landmark index
        
        # Create mask above chin
        h, w = image.shape[:2]
        head_mask = np.zeros((h, w), dtype=np.uint8)
        head_mask[:chin_y+5, :] = 255  # Add small buffer
        
        # Smooth the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, kernel)
        
        # Get full head segmentation
        seg_mask = self.get_head_mask(image)
        
        # Combine masks (intersection)
        final_mask = cv2.bitwise_and(head_mask, seg_mask)
        
        # Apply mask
        result = cv2.bitwise_and(image, image, mask=final_mask)
        return result, final_mask
    
    def get_head_contour(self, image, n_points=500):
        """Extract head contour points"""
        _, mask = self.extract_head_only(image)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        
        # Get largest contour
        cnt = max(contours, key=cv2.contourArea)[:, 0, :]
        
        # Downsample to n_points
        idx = np.round(np.linspace(0, len(cnt)-1, n_points)).astype(int)
        return cnt[idx]


class HeadMorpher:
    """Main class for head morphing operations"""
    
    def __init__(self, segmentation_method='mediapipe'):
        self.segmentor = HeadSegmentor(segmentation_method)
    
    def densify_points(self, pts, max_edge_len=20, n_samples=1):
        """Add points to ensure no edge is too long"""
        pts = pts.copy()
        while True:
            tri = Delaunay(pts)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    a, b = sorted([simplex[i], simplex[(i+1)%3]])
                    edges.add((a, b))
            
            to_add = []
            for i, j in edges:
                p, q = pts[i], pts[j]
                dist = np.linalg.norm(p-q)
                if dist > max_edge_len:
                    for k in range(1, n_samples+1):
                        t = k/(n_samples+1)
                        to_add.append(p*(1-t) + q*t)
            
            if not to_add:
                break
            
            pts = np.vstack([pts, np.array(to_add)])
        
        return pts.astype(np.int32)
    
    def warp_images(self, img1, img2, pts1, pts2, pts_mid):
        """Warp two images using Delaunay triangulation"""
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
    
    def compute_weight_mask(self, shape, mid, margin, axis="x"):
        """Create weight mask for blending"""
        H, W = shape[:2]
        if axis == "x":
            coords = np.arange(W, dtype=np.float32)
            w1d = np.clip((coords - (mid - margin)) / (2*margin), 0.0, 1.0)
            mask = np.tile(w1d, (H, 1))
        else:  # axis=="y"
            coords = np.arange(H, dtype=np.float32)
            w1d = np.clip((coords - (mid - margin)) / (2*margin), 0.0, 1.0)
            mask = np.tile(w1d[:, None], (1, W))
        return cv2.merge([mask, mask, mask])
    
    def synthesize_mid_from_up_down(self, warp_top, warp_bot, pts_mid, 
                                   hairline_margin=20, chin_margin=20):
        """Synthesize middle blend from top/bottom warps"""
        H, W = warp_top.shape[:2]
        all_y = pts_mid[:, 1]
        hairline_y = int(all_y.min())
        chin_y = int(all_y.max())
        
        Y = np.arange(H)[:, None]
        Y2 = np.tile(Y, (1, W))
        
        top_mask = (Y2 < hairline_y + hairline_margin)
        bottom_mask = (Y2 > chin_y - chin_margin)
        middle_mask = ~(top_mask | bottom_mask)
        
        start = hairline_y + hairline_margin
        end = chin_y - chin_margin
        ramp = np.zeros_like(Y2, dtype=np.float32)
        if end > start:
            ramp_vals = (Y2 - start) / float(end - start)
            ramp = np.clip(ramp_vals, 0.0, 1.0)
        Wmask = np.stack([ramp]*3, axis=2)
        
        out = np.zeros_like(warp_top)
        out[top_mask] = warp_top[top_mask]
        out[bottom_mask] = warp_bot[bottom_mask]
        blended = cv2.convertScaleAbs(warp_top*(1-Wmask) + warp_bot*Wmask)
        out[middle_mask] = blended[middle_mask]
        
        return out
    
    def morph_head(self, img1, img2, alpha=0.5, margin=50, axis="x"):
        """Main morphing pipeline"""
        # 1. Extract head regions early
        head1, mask1 = self.segmentor.extract_head_only(img1)
        head2, mask2 = self.segmentor.extract_head_only(img2)
        
        # 2. Get landmarks and contours from head regions
        landmarks1 = self.segmentor.get_face_landmarks(head1)
        landmarks2 = self.segmentor.get_face_landmarks(head2)
        contour1 = self.segmentor.get_head_contour(img1)
        contour2 = self.segmentor.get_head_contour(img2)
        
        if any(x is None for x in [landmarks1, landmarks2, contour1, contour2]):
            raise RuntimeError("Could not extract required features from one or both images")
        
        # 3. Combine points and compute intermediate
        pts1 = np.vstack([landmarks1, contour1])
        pts2 = np.vstack([landmarks2, contour2])
        pts_mid = ((1-alpha)*pts1 + alpha*pts2).astype(np.int32)
        
        # 4. Warp head regions
        warp1, warp2 = self.warp_images(head1, head2, pts1, pts2, pts_mid)
        
        # 5. Blend using the synthesis method
        result = self.synthesize_mid_from_up_down(
            warp1, warp2, pts_mid,
            hairline_margin=20,
            chin_margin=20
        )
        
        # 6. Create final composite with proper masking
        # Use the original image as background
        final = img1.copy()
        
        # Create composite mask
        composite_mask = np.zeros(img1.shape[:2], dtype=np.uint8)
        chin_y = pts_mid[152][1]
        composite_mask[:chin_y+5, :] = 255
        
        # Smooth the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        composite_mask = cv2.morphologyEx(composite_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply result to final image
        final = cv2.bitwise_and(final, final, mask=cv2.bitwise_not(composite_mask))
        final = cv2.add(final, cv2.bitwise_and(result, result, mask=composite_mask))
        
        return final


def compare_generated_to_gt(gen_img, gt_img, segmentor):
    """Compare generated image to ground truth"""
    # Get head mask for GT
    _, gt_mask = segmentor.extract_head_only(gt_img)
    gt_masked = cv2.bitwise_and(gt_img, gt_img, mask=gt_mask)
    
    # Create mask where generated is not black
    mask = np.any(gen_img != 0, axis=2)
    
    if not mask.any():
        raise ValueError("No non-black pixels found in generated image.")
    
    # Crop to bounding box
    ys, xs = np.where(mask)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    gen_crop = gen_img[y1:y2+1, x1:x2+1]
    gt_crop = gt_masked[y1:y2+1, x1:x2+1]
    mask_crop = mask[y1:y2+1, x1:x2+1]
    
    # Compute metrics per channel
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


# Example usage
if __name__ == "__main__":
    # Initialize morpher with different segmentation backends
    morpher = HeadMorpher(segmentation_method='birefnet')  # or 'birefnet', 'rmbg'
    
    # Load images
    imgL = cv2.imread("dataset/1/1/0003.png")
    imgR = cv2.imread("dataset/1/1/0002.png")
    # imgL = cv2.imread("dataset/dataset/BL.jpeg")
    # imgR = cv2.imread("dataset/dataset/BR.jpeg")
    
    # Morph heads
    result = morpher.morph_head(imgL, imgR, alpha=0.25, margin=60, axis="x")
    
    # Display result
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Morphed Head - Optimized Pipeline")
    plt.show()
    
    # Compare with ground truth if available
    # gt = cv2.imread("dataset/1/1/0001.png")
    # metrics = compare_generated_to_gt(result, gt, morpher.segmentor)
    # print(metrics)