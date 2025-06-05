import cv2
import igev_wrapper
import raft_wrapper


def calculate_disparity(left_image, right_image, model):
    if(model == 'IGEV'):
        return igev_wrapper.run_igev_inference(left_image, right_image)
    elif(model == 'RAFT'):
        return raft_wrapper.run_raft_inference(left_image, right_image)
    
    return 

def getVerticalDisparity(top_image, bottom_image, model):
    top_image_rotated = cv2.rotate(top_image, cv2.ROTATE_90_CLOCKWISE)
    bottom_image_rotated = cv2.rotate(bottom_image, cv2.ROTATE_90_CLOCKWISE)

    return calculate_disparity(top_image_rotated, bottom_image_rotated, model)