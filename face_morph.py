import cv2
import time
import numpy as np
from scipy.spatial import Delaunay
import mediapipe as mp
from matplotlib import pyplot as plt


def detect_landmarks(image):
    """Detect facial landmarks using MediaPipe or similar."""
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return np.array([
            [lm.x * image.shape[1], lm.y * image.shape[0]]
            for lm in results.multi_face_landmarks[0].landmark
        ])

def compute_intermediate_landmarks(landmarks_left, landmarks_right):
    """Compute the intermediate landmarks."""
    return (landmarks_left + landmarks_right) / 2

def compute_delaunay_triangulation(points):
    """Compute Delaunay triangulation for given points."""
    return Delaunay(points)

def warp_triangle(img, src_triangle, dst_triangle, dst_img):
    """Warp a single triangle from src_triangle to dst_triangle."""
    # Compute bounding boxes
    src_rect = cv2.boundingRect(np.float32([src_triangle]))
    dst_rect = cv2.boundingRect(np.float32([dst_triangle]))
    
    # Offset triangles by the bounding box
    src_offset_triangle = src_triangle - np.array([src_rect[:2]])
    dst_offset_triangle = dst_triangle - np.array([dst_rect[:2]])
    
    # Get the transformation matrix
    warp_mat = cv2.getAffineTransform(np.float32(src_offset_triangle), np.float32(dst_offset_triangle))
    
    # Warp the source image
    src_cropped = img[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
    dst_cropped = cv2.warpAffine(src_cropped, warp_mat, (dst_rect[2], dst_rect[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    
    # Create a mask for the destination triangle
    mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_offset_triangle), 1, 16, 0)
    
    # Blend the warped triangle into the destination image
    dst_img[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]] = \
        dst_img[dst_rect[1]:dst_rect[1] + dst_rect[3], dst_rect[0]:dst_rect[0] + dst_rect[2]] * (1 - mask[:, :, None]) + \
        dst_cropped * mask[:, :, None]

def warp_images_to_intermediate(left_img, right_img, landmarks_left, landmarks_right, intermediate_landmarks, triangles):
    """Warp the left and right images to the intermediate view."""
    h, w, _ = left_img.shape
    warped_left = np.zeros_like(left_img)
    warped_right = np.zeros_like(right_img)
    
    for triangle in triangles.simplices:
        # Get the triangle vertices
        src_triangle_left = landmarks_left[triangle]
        src_triangle_right = landmarks_right[triangle]
        dst_triangle = intermediate_landmarks[triangle]
        
        # Warp triangles
        warp_triangle(left_img, src_triangle_left, dst_triangle, warped_left)
        warp_triangle(right_img, src_triangle_right, dst_triangle, warped_right)
    
    return warped_left, warped_right

def alpha_blend(warped_left, warped_right):
    """Blend the two warped images."""
    return cv2.addWeighted(warped_left, 0.5, warped_right, 0.5, 0)

def merge_image(foreground, background):
    # Create mask from black pixels
    mask = cv2.threshold(foreground, 1, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Invert mask
    mask_inv = cv2.bitwise_not(mask)
    
    # Get background under the black regions
    bg = cv2.bitwise_and(background, background, mask=mask_inv)
    
    # Get foreground without black regions
    fg = cv2.bitwise_and(foreground, foreground, mask=mask)
    
    # Combine
    result = cv2.add(bg, fg)
    
    return result