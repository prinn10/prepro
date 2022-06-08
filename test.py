import labeling
import cv2
import numpy as np
from PIL import Image

img = cv2.imread('purple_17.png', cv2.IMREAD_COLOR)

# index값을 가진 마스크 이미지 생성
label = np.random.randint(3, size=(161, 586), dtype=np.uint8)
label.fill(0)
label[np.where((img == [1]).all(axis=2))] = [255]

png = Image.fromarray(label).convert('P')

cmap = [[0, 0, 0], [255, 255, 255]]
palette = [value for color in cmap for value in color]
png.putpalette(palette)

print(np.unique(png))  # 라벨 값 확인
png.save('test' + '.png')

# img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# temp = np.zeros(shape=img.shape, dtype=np.uint8)
# temp.fill(0)
# temp[np.where((img == [36, 28, 237]).all(axis=2))] = [0, 0, 255]
# temp[np.where((img == [0, 0, 255]).all(axis=2))] = [0, 0, 255]
# temp[np.where((img == [255, 0, 0]).all(axis=2))] = [0, 0, 255]
# print('convert color')
#
# # 빨강, 검정으로 칠하기
# rows, cols = temp.shape[:2]
# mask = np.zeros((rows + 2, cols + 2), np.uint8)
# newVal = (255, 255, 255)
# loDiff, upDiff = (10, 10, 10), (10, 10, 10)
# retval, temp, mask, rect = cv2.floodFill(temp, mask, (0, 0), newVal, loDiff, upDiff)
# temp[np.where((temp != [255, 255, 255]).all(axis=2))] = [0, 0, 255]
# temp[np.where((temp == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
# temp[np.where((temp == [0, 0, 255]).all(axis=2))] = [255, 255, 255]
#
# # index값을 가진 마스크 이미지 생성
# label = np.random.randint(3, size=(161, 586), dtype=np.uint8)
# label.fill(0)
# label[np.where((temp == [255, 255, 255]).all(axis=2))] = [255]
#
# png = Image.fromarray(label).convert('P')
#
# cmap = [[0, 0, 0], [255, 255, 255]]
# palette = [value for color in cmap for value in color]
# png.putpalette(palette)
#
# png.save(os.path.join(save_path, image_name_without_etx + '.tif'))
# print(np.unique(png))  # 라벨 값 확인
#
