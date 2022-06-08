# 좌우반전 코드
import cv2
import tools
import os

def filp(image, file_name=""):
    return cv2.flip(image, 1)

def grayScale(image, file_name=""):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

#image_full_path, image_name_etx, image_name = tools.image_path('D:\\Total_Dataset\\Dataset Preprocessing\\5.traindata')
image_full_path, image_name, image_name_without_etx = tools.image_path('D:\\Total_Dataset\\Dataset Preprocessing\\4.IndexingMask')
print(image_name)
print(image_name_without_etx)
for i in range(len(image_full_path)):
    image = cv2.imread(image_full_path[i])
    img_gray = filp(image)
    cv2.imwrite(os.path.join('D:\\Total_Dataset\\Dataset Preprocessing\\4.IndexingMask', image_name_without_etx[i]+'_filped.png'), img_gray)

