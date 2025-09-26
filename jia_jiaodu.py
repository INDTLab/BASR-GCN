# 导入数学模块
import math
import os
import shutil
# 定义一个函数，计算两个点之间的角度，返回弧度值
def angle_between_points(p1, p2):
    # 计算两个点的x和y坐标的差值
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # 计算两个点的距离
    distance = math.sqrt(dx**2 + dy**2)
    # 判断距离是否为零，如果是，返回零
    if distance == 0:
        return 0
    # 计算两个点的夹角的正弦值
    sin_angle = dy / distance
    # 计算两个点的夹角的余弦值
    cos_angle = dx / distance
    # 计算两个点的夹角的弧度值，使用math.atan2函数，可以处理正负角度和90度和270度的特殊情况
    radian_angle = math.atan2(sin_angle, cos_angle)
    # 返回弧度值
    return radian_angle

srcpath = r"/data1/jinming/new_datasets/swedish_leaf/swedish_test_skeleton_contour_reduce300/" # 源文件夹路径
dstpath = r"/data1/jinming/new_datasets/swedish_leaf/swedish_test_skeleton_contour_reduce300_jiaodu/" # 目标文件夹路径
if not os.path.exists(dstpath):  # 如果英文名称的子文件夹不存在
    os.mkdir(dstpath)  # 创建英文名称的子文件夹
files = os.listdir(srcpath) # 获取源文件夹下所有txt文件
for file in files:
        # 打开txt文件，读取每一行的内容，存放在一个列表中

        src_file_path=os.path.join(srcpath, file)
        print(src_file_path)
        with open(src_file_path, "r") as f:
            lines = f.readlines()
            points = []
            point_skeleton=[]
            for line in lines:
                # 去掉每一行末尾的换行符
                line = line.strip()
                # 用空格分隔每一行的内容，得到x和y坐标的字符串
                x, y,z,k = line.split()
                # 将x和y坐标转换为浮点数，并存放在一个元组中
                point = (float(x), float(y))

                point_s = (float(z), float(k))
                # 将元组添加到列表中
                points.append(point)
                point_skeleton.append(point_s)

        # 创建一个新的列表，存放每一个点和上一个点的角度值
        angles = []
        # 遍历列表中的每一个点，除了第一个点
        for i in range(1, len(points)):
            # 获取当前点和上一个点
            current_point = points[i]
            previous_point = points[i-1]
            # 调用函数，计算两个点之间的角度值，转换为角度制，并保留两位小数
            angle = round(math.degrees(angle_between_points(previous_point, current_point)), 2)
            # 将角度值添加到列表中
            angles.append(angle)

        # 获取第一个点和最后一个点
        first_point = points[0]
        last_point = points[-1]
        # 调用函数，计算两个点之间的角度值，转换为角度制，并保留两位小数
        angle = round(math.degrees(angle_between_points(last_point, first_point)), 2)
        # 将角度值添加到列表的开头
        angles.insert(0, angle)

        dest_file_path = os.path.join(dstpath, file)
        # 打开一个新的txt文件，写入每一个点的x，y坐标和角度值，用空格分隔
        with open(dest_file_path, "w") as f:
            # 遍历列表中的每一个点，使用一个循环变量i
            for i in range(len(points)):
                # 获取当前点和对应的角度值
                point = points[i]
                angle = angles[i]
                point_s =point_skeleton[i]
                # 将x，y坐标和角度值转换为字符串，并用空格分隔
                line = " ".join([str(point[0]), str(point[1]),str(point_s[0]), str(point_s[1]), str(angle)])
                # 写入文件中，并添加换行符
                f.write(line + "\n")

# 打印完成提示信息
print("Done!")
