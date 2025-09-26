import cv2
import numpy as np
from skimage.morphology import skeletonize, thin
import os

cur_dir = r''
dest = r''
if not os.path.exists(dest):  # 如果英文名称的子文件夹不存在
    os.mkdir(dest)  # 创建英文名称的子文件夹
# if not os.path.exists(dest_picture):  # 如果英文名称的子文件夹不存在
#     os.mkdir(dest_picture)  # 创建英文名称的子文件夹
import numpy as np # 导入 numpy 库


# 遍历当前目录下的所有文件和文件夹
for item in os.listdir(cur_dir):

    file_path = os.path.join(cur_dir, item)

    print('file_path',file_path)

# Load image
    image = cv2.imread(file_path, 0)

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Thinned Skeleton', thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
    ske= thresh
    ske = skeletonize(thresh // 255)
    ske=ske*255
    #print(ske)
    ske = ske.astype(np.uint8)



    coords = []

    for i in range(ske.shape[0]):
        for j in range(ske.shape[1]):
            if ske[i, j] == 255:
                coords.append((j, i))
    coords=np.array(coords)
    #print(coords.shape)
    
    id_gd=np.arange(coords.shape[0])
    id_gd=np.array(id_gd)
    #print(id_gd.shape)
    

    skeleton = coords[id_gd, :]
    #skeleton = coords[id_gd]
    #print(skeleton.shape)


    
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.interpolate import splprep, splev
    

    prefix = item.split('.')[0]
    dest_path=os.path.join(dest, prefix+ '.txt')
    np.savetxt(dest_path, skeleton, delimiter=" ")

    