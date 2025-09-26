import os.path
import cv2 # 用于图像处理
import numpy as np # 用于数学运算
import math
import numpy as np




def reduce_points(points, n):
    # 计算原始坐标点列表中的点数
    num_points = len(points)
    # 如果点数小于等于目标点数n，则无需减少
    if num_points <= n:
        return points
    # 计算步长
    step = math.ceil(num_points / n)


    new_points=[]

    for i in range(0, num_points, step):
        new_points.append(points[i])
    while( len(new_points) > n):
        new_points.pop()
    while( len(new_points) < n):
        for i in range(num_points-10, num_points):
            new_points.append(points[i])
            if len(new_points)==n:
                break
    return np.array(new_points)
# 读取txt文件，第0列为x坐标，第1列为y坐标
file_path =r""
dest_path =r""

if not os.path.exists(dest_path):  # 如果英文名称的子文件夹不存在
    os.mkdir(dest_path)  # 创建英文名称的子文件夹
for file_name in os.listdir(file_path):
    if file_name.endswith(".txt"):
        filename = os.path.join(file_path, file_name)
        #points = np.loadtxt(os.path.join(path,file_name), delimiter='\t')
        points = np.loadtxt(filename, dtype=np.float32)

        #contour= np.array([[x] for x in points])
        #print(type(contour))
        #print(len(contour))

        contour = reduce_points(points,300)


        destname = os.path.join(dest_path, file_name)

        np.savetxt(destname, contour, fmt="%.6f", delimiter=' ')


