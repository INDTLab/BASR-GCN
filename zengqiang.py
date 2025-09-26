

from torchvision.transforms import ToPILImage
import numpy as np
from torchvision import transforms



def fanzhuan(image_path, filename):


    
    image = cv2.imread(image_path)
    
    image = Image.fromarray(image)

    
    image1 =  transforms.functional.rotate(image,30, expand=True)

    image2 = transforms.functional.rotate(image,45, expand=True)

    image3 = transforms.functional.rotate(image,60, expand=True)

    image4 = transforms.functional.rotate(image,75, expand=True)

    image5 = transforms.functional.rotate(image,90, expand=True)

    image6 = transforms.functional.rotate(image,105, expand=True)

    image7 = transforms.functional.rotate(image,120, expand=True)

    image8 = transforms.functional.rotate(image,135, expand=True)

    image9 = transforms.functional.rotate(image,150, expand=True)

    image10 = transforms.functional.rotate(image,165, expand=True)

    image11 = transforms.functional.rotate(image, 180, expand=True)
    image12 = transforms.functional.rotate(image, 195, expand=True)
    image13 = transforms.functional.rotate(image, 210, expand=True)
    image14 = transforms.functional.rotate(image, 225, expand=True)
    image15 = transforms.functional.rotate(image, 240, expand=True)
    image16 = transforms.functional.rotate(image, 255, expand=True)
    image17 = transforms.functional.rotate(image, 270, expand=True)
    image18 = transforms.functional.rotate(image, 285, expand=True)
    image19 = transforms.functional.rotate(image, 300, expand=True)
    image20 = transforms.functional.rotate(image, 315, expand=True)

   
    image1 = np.array(image1)
    image2 = np.array(image2)
    image3 = np.array(image3)
    image4 = np.array(image4)
    image5 = np.array(image5)
    image6 = np.array(image6)
    image7 = np.array(image7)
    image8 = np.array(image8)
    image9 = np.array(image9)
    image10 = np.array(image10)
    image11 = np.array(image11)
    image12 = np.array(image12)
    image13 = np.array(image13)
    image14 = np.array(image14)
    image15 = np.array(image15)
    image16 = np.array(image16)
    image17 = np.array(image17)
    image18 = np.array(image18)
    image19 = np.array(image19)
    image20 = np.array(image20)
    cv2.imwrite(filename + "-1.png", image1)
    cv2.imwrite(filename + "-2.png", image2)
    cv2.imwrite(filename + "-3.png", image3)
    cv2.imwrite(filename + "-4.png", image4)
    cv2.imwrite(filename + "-5.png", image5)
    cv2.imwrite(filename + "-6.png", image6)
    cv2.imwrite(filename + "-7.png", image7)
    cv2.imwrite(filename + "-8.png", image8)
    cv2.imwrite(filename + "-9.png", image9)
    cv2.imwrite(filename + "-10.png", image10)

    cv2.imwrite(filename + "-11.png", image11)
    cv2.imwrite(filename + "-12.png", image12)
    cv2.imwrite(filename + "-13.png", image13)
    cv2.imwrite(filename + "-14.png", image14)
    cv2.imwrite(filename + "-15.png", image15)
    cv2.imwrite(filename + "-16.png", image16)
    cv2.imwrite(filename + "-17.png", image17)
    cv2.imwrite(filename + "-18.png", image18)
    cv2.imwrite(filename + "-19.png", image19)
    cv2.imwrite(filename + "-20.png", image20)

from PIL import Image


import os
import shutil

import cv2
import re
cur_dir = r''

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', s)]
# 指定要移动的子文件夹的名称列表
L = os.listdir(cur_dir)


dest = r''

if not os.path.exists(dest):
    os.makedirs(dest)
# 遍历当前目录下的所有文件和文件夹
for item in os.listdir(cur_dir):
    # 如果是要移动的子文件夹，就把它下面的所有文件移动到目标文件夹
    if item in L:

        sub_dir = os.path.join(cur_dir, item)
        des_dir = os.path.join(dest, item)
        if not os.path.exists(des_dir):
            os.makedirs(des_dir)
        files = os.listdir(sub_dir)
        #files.sort(key=natural_sort_key)

        for file in files:
         if file.endswith(".png"):
           file_path = os.path.join(sub_dir, file)

           des_path = os.path.join(des_dir, file)
           file_path1 =des_path.split(".")[0]

           #print(des_path)
           fanzhuan(file_path,file_path1)


