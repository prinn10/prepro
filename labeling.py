
from PIL import Image

import numpy as np # linear algebra

import os, cv2
import tools

def createTwoMask(image_path, image_name_without_etx, save_path_png, save_path_tif):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    temp = np.zeros(shape=img.shape, dtype=np.uint8)
    temp.fill(0)

    # temp[np.where((img == [36, 28, 237]).all(axis=2))] = [0, 0, 255]
    # temp[np.where((img == [52, 50, 222]).all(axis=2))] = [0, 0, 255]
    # temp[np.where((img == [0, 0, 255]).all(axis=2))] = [0, 0, 255]
    # temp[np.where((img == [255, 0, 0]).all(axis=2))] = [0, 0, 255]

    h, w, c = img.shape
    for i in range(h):
        for j in range(w):
            if img[i][j][0] >= 20 and img[i][j][0] <= 68 \
                    and img[i][j][1] >= 20 and img[i][j][1] <= 68 \
                    and img[i][j][2] >= 190:
                temp[i][j] = [0, 0, 255]

    print('convert color')

    # 빨강, 검정으로 칠하기
    rows, cols = temp.shape[:2]
    mask = np.zeros((rows + 2, cols + 2), np.uint8)
    newVal = (255, 255, 255)
    loDiff, upDiff = (10, 10, 10), (10, 10, 10)
    retval, temp, mask, rect = cv2.floodFill(temp, mask, (0, 0), newVal, loDiff, upDiff)
    temp[np.where((temp != [255, 255, 255]).all(axis=2))] = [0, 0, 255]
    temp[np.where((temp == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
    temp[np.where((temp == [0, 0, 255]).all(axis=2))] = [255, 255, 255]

    #index값을 가진 마스크 이미지 생성
    label = np.random.randint(3, size=(144, 586), dtype=np.uint8)
    label.fill(0)
    label[np.where((temp == [255, 255, 255]).all(axis=2))] = [1]

    png = Image.fromarray(label).convert('P')

    cmap = [[0, 0, 0], [255, 255, 255]]
    palette = [value for color in cmap for value in color]

    png.putpalette(palette)

    # Labels 이미지 저장
    png.save(os.path.join(save_path_tif, image_name_without_etx+'.tif')) # tif class 0,1
    png.save(os.path.join(save_path_png, image_name_without_etx+'.png')) # png class 0, 255


def createMask(image_path, image_name_without_etx, save_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    temp = np.zeros(shape=img.shape, dtype=np.uint8)
    temp.fill(0)
    temp[np.where((img == [36, 28, 237]).all(axis=2))] = [0, 0, 255]
    temp[np.where((img == [0, 0, 255]).all(axis=2))] = [0, 0, 255]
    temp[np.where((img == [255, 0, 0]).all(axis=2))] = [0, 0, 255]
    print('convert color')

    # 빨강, 검정으로 칠하기
    rows, cols = temp.shape[:2]
    mask = np.zeros((rows + 2, cols + 2), np.uint8)
    newVal = (255, 255, 255)
    loDiff, upDiff = (10, 10, 10), (10, 10, 10)
    retval, temp, mask, rect = cv2.floodFill(temp, mask, (0, 0), newVal, loDiff, upDiff)
    temp[np.where((temp != [255, 255, 255]).all(axis=2))] = [0, 0, 255]
    temp[np.where((temp == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
    temp[np.where((temp == [0, 0, 255]).all(axis=2))] = [255, 255, 255]

    #index값을 가진 마스크 이미지 생성
    label = np.random.randint(3, size=(144, 586), dtype=np.uint8)
    label.fill(0)
    label[np.where((temp == [255, 255, 255]).all(axis=2))] = [255]

    png = Image.fromarray(label).convert('P')

    cmap = [[0, 0, 0], [255, 255, 255]]
    palette = [value for color in cmap for value in color]
    png.putpalette(palette)

    png.save(os.path.join(save_path, image_name_without_etx + '.tif'))
    print(np.unique(png))  # 라벨 값 확인

def MaskPutPallet(image_path, image_name_without_etx, save_path): #까만 png 인덱스 이미지에 색칠하여 저장하는 함수
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    #index값을 가진 마스크 이미지 생성
    label = np.random.randint(3, size=(161, 586), dtype=np.uint8)
    label.fill(0)
    label[np.where((img == [1]).all(axis=2))] = [255]

    png = Image.fromarray(label).convert('P')

    cmap = [[0, 0, 0], [255, 255, 255]]
    palette = [value for color in cmap for value in color]
    png.putpalette(palette)

    print(np.unique(png))  # 라벨 값 확인
    png.save(os.path.join(save_path, image_name_without_etx + '.tif'))

def datasetInspection(dataset_path = 'D:\\Total_Dataset\\real_dataset'): # 확장자, 이미지 무결성 검토
    label_path = os.path.join(dataset_path, 'label')
    test_path = os.path.join(dataset_path, 'test')
    train_path = os.path.join(dataset_path, 'train')


    label_full_path, label_name, label_name_without_etx = tools.image_path(label_path)
    test_full_path, test_name, test_name_without_etx = tools.image_path(test_path)
    train_full_path, train_name, train_name_without_etx = tools.image_path(train_path)

    # 이미지 개수 출력
    print('label count :',len(os.listdir(label_path)))
    print('test count :',len(os.listdir(test_path)))
    print('train count :',len(os.listdir(train_path)))

    #라벨 이미지 인덱스 값 확인
    for label in label_full_path:
        print(label,np.unique(cv2.imread(label)))  # 라벨 값 확인

    #확장자 무결성 검사(이름만 png이고 다른 확장자의 형식을 띄는지 검증) True면 정상 False면 비정상
    error = 0
    for label in label_full_path:
        # print(str(is_image(label, True)))
        if tools.is_image(label, False) == False:
            print(label)
            error += 1
    print('label data 총', len(label_full_path), '개 정상:', len(label_full_path) - error, ' 에러:', error)


    # test
    error = 0
    error_test_path_list = []
    for test in test_full_path:
        # print(str(is_image(label, True)))
        if tools.is_image(test, False) == False:
            error_test_path_list.append(test)
            print(test)
            error += 1
    print('test data 총', len(test_full_path), '개 정상:', len(test_full_path) - error, ' 에러:', error)
    if error > 0:
        user_input = input('에러 데이터를 정상으로 변환하시겠습니까?')
        if user_input == 'y':
            if error > 0: # error 이미지 변환
                for error_test_path in error_test_path_list:
                    im1 = Image.open(error_test_path)
                    im1.save(error_test_path)

        error = 0 # dataset 검증
        error_train_path_list = []
        for test in test_full_path:
            # print(str(is_image(label, True)))
            if tools.is_image(test, False) == False:
                error_train_path_list.append(test)
                print(test)
                error += 1
        print('test data 총', len(test_full_path), '개 정상:', len(test_full_path) - error, ' 에러:', error)

    # train
    error = 0
    error_train_path_list = []
    for train in train_full_path:
        # print(str(is_image(label, True)))
        if tools.is_image(train, False) == False:
            error_train_path_list.append(train)
            print(train)
            error += 1
    print('train data 총', len(train_full_path), '개 정상:', len(train_full_path) - error, ' 에러:', error)
    if error > 0:
        user_input = input('에러 데이터를 정상으로 변환하시겠습니까?')
        if user_input == 'y':
            if error > 0: # error 이미지 변환
                for error_train_path in error_train_path_list:
                    im1 = Image.open(error_train_path)
                    im1.save(error_train_path)

        error = 0 # dataset 검증
        error_train_path_list = []
        for train in train_full_path:
            # print(str(is_image(label, True)))
            if tools.is_image(train, False) == False:
                error_train_path_list.append(train)
                print(train)
                error += 1
        print('train data 총', len(train_full_path), '개 정상:', len(train_full_path) - error, ' 에러:', error)


# dir_name = '2011028_OK'
#
# labeling_path = os.path.join('D:\\Total_Dataset\\Dataset\\3. Handmaking Labeling Image', dir_name + '\\wire_barrel_front')
# origin_path = os.path.join('D:\\Total_Dataset\\Dataset\\2. Crop Origin Image', dir_name + '\\wire_barrel_front')
#
# image_full_path, image_name, image_name_without_etx = tools.image_path(labeling_path)
#
# for i in range(len(image_full_path)):
#     createTwoMask(image_full_path[i],image_name_without_etx[i],'D:\\Total_Dataset\\Dataset\\5. Segmentation_dataset\\Wire_front\\No_Augmentation\\png\\train_labels','D:\\Total_Dataset\\Dataset\\5. Segmentation_dataset\\Wire_front\\No_Augmentation\\tiff\\train_labels')
#     tools.convertEtx('tiff', os.path.join(origin_path, image_name[i]), image_name_without_etx[i], 'D:\\Total_Dataset\\Dataset\\5. Segmentation_dataset\\Wire_front\\No_Augmentation\\tiff\\train')
#     tools.convertEtx('png', os.path.join(origin_path, image_name[i]), image_name_without_etx[i], 'D:\\Total_Dataset\\Dataset\\5. Segmentation_dataset\\Wire_front\\No_Augmentation\\png\\train')

if __name__ == '__main__':
    #test 파일 생성
    image_full_path, image_name, image_name_without_etx = tools.image_path('D:\\Total_Dataset\\Dataset\\2. Crop Origin Image\\2011028_OK\\wire_barrel_front')

    for i in range(len(image_full_path)):
        tools.convertEtx('tiff', image_full_path[i], image_name_without_etx[i], 'D:\\Total_Dataset\\Dataset\\6. Unlabeld_dataset')