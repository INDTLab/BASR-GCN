#遍历文件夹和子文件下的文件并操作

import numpy as np
from PIL import Image, ImageOps
import cv2
# Open the image file
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

# 获取目标文件夹的路径
#dest = os.path.join(cur_dir, 'target_folder')
dest = r'D:\shujvji\ETH-80-master\ETH-80\heibai'
if not os.path.exists(dest):  # 如果英文名称的子文件夹不存在
    os.mkdir(dest)  # 创建英文名称的子文件夹
chakan = r'D:\shujvji\shape\Mpeg7\Mpeg7\MPEG-fanzhuanzengqiang\chakan_test'
if not os.path.exists(chakan):  # 如果英文名称的子文件夹不存在
    os.mkdir(chakan)  # 创建英文名称的子文件夹
# 遍历当前目录下的所有文件和文件夹
for item in os.listdir(cur_dir):
    
    


            
          if file.endswith(".png"):
             file_path = os.path.join(sub_dir, file)
         
             dest_path = os.path.join(des_dir, file)
             #print('file_path',dest_path)
         
             img = cv2.imread(file_path)
         
             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度模式
             ret, thresh = cv2.threshold(img, 238, 255, cv2.THRESH_BINARY)  # 将图像二值化，只保留白色和黑色
             contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 找到图像中的白色轮廓
             cv2.drawContours(img, contours, -1, (255, 255, 255), -1)  # 填充轮廓内部的黑色像素点为白色
         
             thresh = cv2.bitwise_not(thresh)
             cv2.imwrite(dest_path, thresh)

            


