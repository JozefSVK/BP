import utilities
import cv2
from matplotlib import pyplot as plt
import numpy as np
import view_synthesis

def func4(top_left_image_path, top_right_image_path, bottom_left_image_path, bottom_right_image_path, downscale=0.5, model='IGEV', alpha=0.5, beta=0.5):
    TL_image = cv2.imread(top_left_image_path)
    TR_image = cv2.imread(top_right_image_path)
    BL_image = cv2.imread(bottom_left_image_path)
    BR_image = cv2.imread(bottom_right_image_path)



    # downscale
    resized_images = [cv2.resize(image, None, fx=downscale, fy=downscale) for image in [TL_image, TR_image, BL_image, BR_image]]
    TL_image, TR_image, BL_image, BR_image = resized_images

    # horizontal disparity
    TL_disparityH, TR_disparityH = utilities.calculate_disparity(TL_image, TR_image, model)
    BL_disparityH, BR_disparityH = utilities.calculate_disparity(BL_image, BR_image, model)
    print("Horizontal disparity done")

    # vertical disparity
    BL_disparityV, TL_disparityV = utilities.getVerticalDisparity(BL_image, TL_image, model)
    BR_disparityV, TR_disparityV = utilities.getVerticalDisparity(BR_image, TR_image, model)
    print("Vertical disparity done")

    alpha = 1 - alpha
    beta = 1 - beta

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files
    # appendFilename = "4cameras"
    # filename = f"img_{model}_{appendFilename}.mp4"
    # height, width = TL_image.shape[:2]
    # video_writer = cv2.VideoWriter(filename, fourcc, 10, (width, height))
    # inter_values = np.linspace(0, 1, 11)
    # for valueAlpha in inter_values:
    #     for valueBeta in inter_values:
    #         valueAlpha = round(valueAlpha, 2)
    #         valueBeta = round(valueBeta, 2)
    #         print("value alpha " + str(valueAlpha) + " value beta " + str(valueBeta))
    #         imgI = create_intermediate_view([TL_image, TL_disparityH, TL_disparityV], [TR_image, TR_disparityH, TR_disparityV], 
    #                                          [BL_image, BL_disparityH, BL_disparityV], [BR_image, BR_disparityH, BR_disparityV],
    #                                          valueAlpha, valueBeta, model)
    #         imgI = cv2.cvtColor(imgI, cv2.COLOR_BGR2RGB)
    #         video_writer.write(imgI)
    # video_writer.release()

    imgI = create_intermediate_view([TL_image, TL_disparityH, TL_disparityV], [TR_image, TR_disparityH, TR_disparityV], 
                                             [BL_image, BL_disparityH, BL_disparityV], [BR_image, BR_disparityH, BR_disparityV],
                                             alpha, beta, model)
    imgI = cv2.cvtColor(imgI, cv2.COLOR_BGR2RGB)
    cv2.imwrite("middle.png", imgI)


intermediate_cache = {}

def synthesis_view(left_list, right_list, alpha, top_down = False):
    imgL = left_list[0]
    disparityLR = left_list[1]

    imgR = right_list[0]
    disparityRL = right_list[1]

    if (top_down):
        imgL = cv2.rotate(imgL, cv2.ROTATE_90_CLOCKWISE)
        imgR = cv2.rotate(imgR, cv2.ROTATE_90_CLOCKWISE)

        disparityLR = left_list[2]
        disparityRL = right_list[2]

    imgI, disparityIL, disparityIR, imgIL, imgIR  = view_synthesis.create_intermediate_view(imgL, imgR, disparityLR, disparityRL, alpha)
    
    if top_down:
        imgI = cv2.rotate(imgI, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return imgI


# the list will contain [0] the image, [1] the horizontal disparity map, [2] vertical disparity map
def create_intermediate_view(top_left_list, top_right_list, bottom_left_list, bottom_right_list, alpha = 0.5, beta = 0.5, model = 'IGEV'):
    if any(len(lst) != 3 for lst in [top_left_list, top_right_list, bottom_left_list, bottom_right_list]):
        raise Exception("Some array in create intermediate view doesn't contain 3 items")

    not_in_cache_alpha = False
    not_in_cache_beta = False

    # horizontal view synthesis
    if((round(alpha, 2), round(0.0, 2)) in intermediate_cache):
        top_image = intermediate_cache[(round(alpha, 2), round(0.0, 2))]
    else:
        top_image = synthesis_view(top_left_list, top_right_list, alpha)
        not_in_cache_alpha = True

    if((round(alpha, 2), round(1.0, 2)) in intermediate_cache):
        bottom_image = intermediate_cache[(round(alpha, 2), round(1.0, 2))]
    else:
        bottom_image = synthesis_view(bottom_left_list, bottom_right_list, alpha)
        not_in_cache_alpha = True
    

    # vertical view synthesis\
    if((round(0.0, 2), round(beta, 2)) in intermediate_cache):
        left_image = intermediate_cache[(round(0.0, 2), round(beta, 2))]
    else:
        left_image = synthesis_view(bottom_left_list, top_left_list, beta, True)
        not_in_cache_beta = True

    # left_image = synthesis_view(bottom_left_list, top_left_list, beta, True)
    if((round(1.0, 2), round(beta, 2)) in intermediate_cache):
        right_image = intermediate_cache[(round(1.0, 2), round(beta, 2))]
    else:
        right_image = synthesis_view(bottom_right_list, top_right_list, beta, True)
        not_in_cache_beta = True
    # right_image = synthesis_view(bottom_right_list, top_right_list, beta, True)

    # plt.subplot(2, 2, 1)  # (rows, columns, index) 
    # plt.imshow(top_image)
    # plt.title('top image')
    # plt.axis('off')

    # plt.subplot(2, 2, 2)  # (rows, columns, index) 
    # plt.imshow(bottom_image)
    # plt.title('bottom image')
    # plt.axis('off')

    # plt.subplot(2, 2, 3)  # (rows, columns, index) 
    # plt.imshow(left_image)
    # plt.title('left image')
    # plt.axis('off')

    # plt.subplot(2, 2, 4)  # (rows, columns, index) 
    # plt.imshow(right_image)
    # plt.title('right image')
    # plt.axis('off')

    # plt.show()

    if not_in_cache_alpha:
        disparityBT, disparityTB = utilities.getVerticalDisparity(bottom_image, top_image, model)
    else:
        disparityBT = bottom_image[2]
        disparityTB = top_image[2]

        bottom_image = bottom_image[0]
        top_image = top_image[0]

    if not_in_cache_beta:
        disparityLR, disparityRL = utilities.calculate_disparity(left_image, right_image, model)
    else:
        disparityLR = left_image[1]
        disparityRL = right_image[1]

        left_image = left_image[0]
        right_image = right_image[0]

    imgIV = synthesis_view([bottom_image, None, disparityBT], [top_image, None, disparityTB], beta, True)
    imgIH = synthesis_view([left_image, disparityLR, None], [right_image, disparityRL, None], alpha)

    if (not_in_cache_alpha or not_in_cache_beta) and not ((alpha == 0.0 or alpha == 1.0) and (beta == 0.0 or beta == 1.0)):
        print("Added image ", alpha, beta)
        intermediate_cache[(round(alpha, 2), round(0.0, 2))] = [top_image, None, disparityTB]
        intermediate_cache[(round(alpha, 2), round(1.0, 2))] = [bottom_image, None, disparityBT]
        intermediate_cache[(round(0.0, 2), round(beta, 2))] = [left_image, disparityLR, None]
        intermediate_cache[(round(1.0, 2), round(beta, 2))] = [right_image, disparityRL, None]

    imgIV = cv2.cvtColor(imgIV, cv2.COLOR_BGR2RGB)
    imgIH = cv2.cvtColor(imgIH, cv2.COLOR_BGR2RGB)

    final_image = cv2.addWeighted(imgIV, 0.5, imgIH, 0.5, 0.0)

    # plt.subplot(3, 1, 1)  # (rows, columns, index) 
    # plt.imshow(imgIV)
    # plt.title('Vertical middle')
    # plt.axis('off')

    # plt.subplot(3, 1, 2)  # (rows, columns, index) 
    # plt.imshow(imgIH)
    # plt.title('Horizontal middle')
    # plt.axis('off')

    # plt.subplot(3, 1, 3)  # (rows, columns, index) 
    # plt.imshow(final_image)
    # plt.title('bleded middle')
    # plt.axis('off')

    # plt.show()

    # final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("dataset/custom_res/Middle_view.png", final_image)

    return final_image




if __name__ == "__main__":
    model_name = "IGEV"

    dataset_path = "dataset/dataset/"
    TL_image_path = dataset_path + "TL.jpeg"
    TR_image_path = dataset_path + "TR.jpeg"
    BL_image_path = dataset_path + "BL.jpeg"
    BR_image_path = dataset_path + "BR.jpeg"

    func4(TL_image_path, TR_image_path, BL_image_path, BR_image_path, 1/3, model_name, 0.5, 0.5)