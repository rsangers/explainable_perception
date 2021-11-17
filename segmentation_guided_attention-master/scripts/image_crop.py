import os
import cv2

def image_clean(img_path, input_dir, output_dir):
    img = cv2.imread(input_dir+img_path)
    crop_img = img[0:340, 0:480]
    cv2.imwrite(output_dir+img_path, crop_img )


if __name__ == '__main__':
    IMAGES_PATH= 'D:/Users/Ruben/Downloads/placepulse_images/'
    CROPPED_PATH = 'D:/Users/Ruben/Downloads/segmentation_guided_attention-master/segmentation_guided_attention-master/scripts/pp_cropped/'
    img_list = os.listdir(IMAGES_PATH)
    count = 0
    total = len(img_list)
    for img_path in img_list:
        count += 1
        print(f'{count}/{total}\r', end="")
        image_clean(img_path,IMAGES_PATH, CROPPED_PATH)  
