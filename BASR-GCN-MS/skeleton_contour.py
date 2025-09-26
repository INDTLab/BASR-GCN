import numpy as np
import os.path
import cv2 # 用于图像处理
import numpy as np # 用于数学运算
import math
import numpy as np

import numpy as np
#欧式距离法计算
def find_corresponding_skeleton_point(contour_file, skeleton_file, output_file):
    # 读取轮廓点和骨架点
    contour_points = np.loadtxt(contour_file, dtype=np.float32)
    skeleton_points = np.loadtxt(skeleton_file, dtype=np.float32)
    contour_points = np.array(contour_points)
    skeleton_points = np.array(skeleton_points)
    # 找到对应的骨架点
    corresponding_points = []
    for contour_point in contour_points:
        #distances = np.sqrt(np.sum((skeleton_points - contour_point)**2, axis=1))
        distances = np.sum(np.abs(skeleton_points - contour_point), axis=1)
        corresponding_point = skeleton_points[np.argmin(distances)]

        

    # 写入到输出文件
    np.savetxt(output_file, corresponding_points,fmt="%.6f", delimiter=' ')


import numpy as np
import networkx as nx

def find_corresponding_skeleton_point_graph(contour_file, skeleton_file, output_file):
    contour_points = np.loadtxt(contour_file, dtype=np.float32)
    skeleton_points = np.loadtxt(skeleton_file, dtype=np.float32)
    contour_points = np.array(contour_points)
    skeleton_points = np.array(skeleton_points)
    # 构建图形
    G = nx.Graph()
    for i, contour_point in enumerate(contour_points):
        for j, skeleton_point in enumerate(skeleton_points):
            distance = np.linalg.norm(np.array(contour_point) - np.array(skeleton_point))
            G.add_edge(i, len(contour_points) + j, weight=distance)

    # 找到最短路径
    corresponding_points = []
    for i in range(len(contour_points)):
        path = nx.shortest_path(G, i, len(contour_points), weight='weight')
        corresponding_point = skeleton_points[path[1] - len(contour_points)]
        corresponding_points.append(corresponding_point)

    np.savetxt(output_file, corresponding_points, fmt="%.6f", delimiter=' ')
    #return corresponding_points



# 测试
#find_corresponding_skeleton_point('contour.txt', 'skeleton.txt', 'output.txt')


# 测试
#find_corresponding_skeleton_point('contour.txt', 'skeleton.txt', 'output.txt')

contour_file_path =r""


skeleton_file_path = r""

dest_path =r""

if not os.path.exists(dest_path):  # 如果英文名称的子文件夹不存在
    os.mkdir(dest_path)  # 创建英文名称的子文件夹
for file_name in os.listdir(contour_file_path):
    if file_name.endswith(".txt"):
        contour_file = os.path.join(contour_file_path, file_name)
        #print(contour_file)
        skeleton_file =os.path.join(skeleton_file_path, file_name)
        output_file=os.path.join(dest_path, file_name)
        print(output_file)
        #find_corresponding_skeleton_point(contour_file, skeleton_file, output_file)
        find_corresponding_skeleton_point(contour_file, skeleton_file, output_file)







import numpy as np
import cv2
#切线法计算
def find_corresponding_skeleton_point_qiexian(contour_file, skeleton_file,output_file):
    # 创建一个空的黑色图像
    img = np.zeros((1200, 1200), dtype=np.uint8)
    contour_points = np.loadtxt(contour_file, dtype=np.float32)
    skeleton_points = np.loadtxt(skeleton_file, dtype=np.float32)
    contour_points = np.array(contour_points)
    skeleton_points = np.array(skeleton_points)
    # 在图像上画出轮廓点
    for point in contour_points:
        #print(point[1])
        #print(point[0])
        img[int(point[1]), int(point[0])] = 255

    # 计算轮廓点的切线方向
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    gradient = np.arctan2(sobely, sobelx)

    corresponding_points = []
    for point in contour_points:
        # 计算法向量的方向
        normal_direction = gradient[int(point[1]), int(point[0])] + np.pi/2

        # 找到与法向量方向最接近的骨架点
        min_distance = float('inf')
        corresponding_point = None
        for skeleton_point in skeleton_points:
            vector = np.array(skeleton_point) - np.array(point)
            direction = np.arctan2(vector[1], vector[0])
            distance = np.linalg.norm(vector)
            if abs(direction - normal_direction) < np.pi/2 and distance < min_distance:
                min_distance = distance
                corresponding_point = skeleton_point
        if corresponding_point is None:
            distances = np.sqrt(np.sum((skeleton_points - point) ** 2, axis=1))
            corresponding_point = skeleton_points[np.argmin(distances)]

        #corresponding_point=np.array(corresponding_point)
        #corresponding_points.append(corresponding_point)
        #print(point.shape)
        #print(corresponding_point.shape)
        corresponding_points.append(np.concatenate((point, corresponding_point)))

    np.savetxt(output_file, corresponding_points, fmt="%.6f", delimiter=' ')

    #return corresponding_points