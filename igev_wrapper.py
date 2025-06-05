import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from IGEV_plusplus.core.igev_stereo import IGEVStereo
from IGEV_plusplus.core_rt.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2
from IGEV_plusplus.core_rt.utils.frame_utils import readPFM


DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def run_igev_inference(left_image, right_image):
    # Create temporary args object with the same attributes as ArgumentParser
    class Args:
        def __init__(self):
            self.restore_ckpt = './IGEV_plusplus/models/middlebury.pth'
            self.save_numpy = True
            self.mixed_precision = False
            self.precision_dtype = 'float32'
            self.valid_iters = 64
            self.hidden_dims = [128]*3
            self.corr_levels = 2
            self.corr_radius = 4
            self.n_downsample = 2
            self.n_gru_layers = 3
            self.max_disp = 768
            self.s_disp_range = 48
            self.m_disp_range = 96
            self.l_disp_range = 192
            self.s_disp_interval = 1
            self.m_disp_interval = 2
            self.l_disp_interval = 4
    
    args = Args()
    return demo(args, left_image, right_image)


def prepare_image(img):
    img = np.array(img).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def demo(args, left_image, right_image):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    left_image_flipped = cv2.flip(left_image, 1)
    right_image_flipped = cv2.flip(right_image, 1)

    def compute_disparity(left, right):
        image1 = prepare_image(left.copy())
        image2 = prepare_image(right.copy())

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
        disp = padder.unpad(disp)
        disp = disp.cpu().numpy().squeeze()

        return disp

    with torch.no_grad():
        disparityLR = compute_disparity(left_image, right_image)
        disparityRL = compute_disparity(right_image_flipped, left_image_flipped)
        return disparityLR, cv2.flip(disparityRL, 1)
