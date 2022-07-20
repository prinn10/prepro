
from PIL import Image

import numpy as np # linear algebra

import os, cv2
import tools
import shutil

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

def original_size_mask(image_path, image_name_without_etx, save_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    temp = np.zeros(shape=img.shape, dtype=np.uint8)
    temp.fill(0)
    bgr_color_list = [[36, 28, 237], [204, 72, 63], [164, 73, 163], [76, 177, 34], [39, 127, 255]]
    rgb_color_list = [[237, 28, 36], [63, 72, 204], [163, 73, 164], [34, 177, 76], [255, 127, 39]]

    for color, rgb_color in zip(bgr_color_list, rgb_color_list):
        temp[np.where((img == color).all(axis=2))] = rgb_color

    print(np.unique(temp))
    print('original size convert color')

    im = Image.fromarray(temp)
    im.save(os.path.join('D:\\temp', image_name_without_etx + '.bmp'))

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

def dataset_rename(dir_path, date_dir_list, part_dir_list):
    for date_dir in date_dir_list:
        for part_dir in part_dir_list:
            target_path = os.path.join(dir_path, date_dir)
            target_path = os.path.join(target_path, part_dir)
            dst_name_list = []
            image_full_path_list, image_name_list, image_name_without_etx_list = tools.image_path(target_path)
            for image_name in image_name_list:
                dst_name_list.append(os.path.join(target_path, (date_dir + '_' + image_name)))
            src_name_list = image_full_path_list

            for src_name, dst_name in zip(src_name_list, dst_name_list):
                print('rename', src_name, dst_name)
                rename(src_name, dst_name)

def create_mask_dataset(src_base_path, dst_base_path, date_dir_list, part_dir_list):
    ori_base_path = 'D:\\Total_Dataset\\Dataset\\ori_temp'
    for date_dir in date_dir_list:
        for part_dir in part_dir_list:
            labled_data_count = 0
            ori_dir_path = os.path.join(ori_base_path, date_dir)
            ori_dir_path = os.path.join(ori_dir_path, part_dir)
            src_dir_path = os.path.join(src_base_path, date_dir)
            src_dir_path = os.path.join(src_dir_path, part_dir)
            dst_dir_path = os.path.join(dst_base_path, part_dir)

            image_full_path_list, image_name_list, image_name_without_etx_list = tools.image_path(src_dir_path)
            for i in range(len(image_full_path_list)):
                if advanced_create_mask(image_full_path_list[i], os.path.join(dst_dir_path, ('train_lables' + '\\' + image_name_without_etx_list[i] + '.tif'))) == True: # lable 이미지 저장
                    shutil.copy(os.path.join(ori_dir_path, image_name_list[i]), os.path.join(dst_dir_path, ('train' + '\\' + image_name_list[i]))) # origin img 이동
                    labled_data_count += 1

            print(part_dir,date_dir,'total:', len(image_full_path_list), 'labled_data:', labled_data_count)

# def advanced_create_mask(image_full_path):
#     # 1. Handmaking Image load
#     img = cv2.imread(image_full_path, cv2.IMREAD_COLOR)
#
#     # 2. GT img Instance 생성, lable color를 제외한 모든 pixcel을 blob color로 변환
#     lable_color_list = [[36, 28, 237], [0, 0, 255]]
#     blob_color = [125, 125, 125]
#     GT_img = np.zeros(shape=img.shape, dtype=np.uint8)
#     # 핸드라벨링 컬러 > 리얼라벨링 컬러
#     for lable_color in lable_color_list:
#         GT_img[np.where((img == lable_color).all(axis=2))] = [255, 255, 255]
#     # 이외에는 blob으로
#     GT_img[np.where((GT_img != [255, 255, 255]).all(axis=2))] = blob_color
#
#
#     # 3. Fill background color
#     background_color = (0, 0, 0)
#     rows, cols = GT_img.shape[:2]
#     mask = np.zeros((rows + 2, cols + 2), np.uint8)
#     retval, GT_img, mask, rect = cv2.floodFill(GT_img, mask, (0, 0), background_color)
#
#     # 4. blob color > Labeling color로 변환
#     GT_img[np.where((GT_img == blob_color).all(axis=2))] = [255,255,255]
#
#     # 5. index값을 가진 마스크 이미지 생성
#     label = np.random.randint(3, size=(rows, cols), dtype=np.uint8)
#     label.fill(0)
#     label[np.where((GT_img == [255, 255, 255]).all(axis=2))] = [1]
#     png = Image.fromarray(label).convert('P')
#     cmap = [[0, 0, 0], [255, 255, 255]]
#     palette = [value for color in cmap for value in color]
#     png.putpalette(palette)
#     png.save('test.tif')

def advanced_create_mask(image_full_path, dst_full_path):
    # 1. Handmaking Image load
    img = cv2.imread(image_full_path, cv2.IMREAD_COLOR)

    # 2. GT img Instance 생성, lable color를 제외한 모든 pixcel을 blob color로 변환
    lable_color_list = [[36, 28, 237], [0, 0, 255], [204, 72, 63]]
    blob_color = [125, 125, 125]
    GT_blue_img = np.zeros(shape=img.shape, dtype=np.uint8)
    # 핸드라벨링 컬러 > 리얼라벨링 컬러
    for lable_color in lable_color_list:
        GT_blue_img[np.where((img == lable_color).all(axis=2))] = [30, 30, 30]
    # cv2.imshow('test', GT_blue_img)
    # cv2.waitKey(0)

    # 이외에는 blob으로
    GT_blue_img[np.where((GT_blue_img != [30, 30, 30]).all(axis=2))] = blob_color

    # cv2.imshow('test', GT_blue_img)
    # cv2.waitKey(0)

    GT_red_img = np.zeros(shape=img.shape, dtype=np.uint8)
    lable_color_list = [[36, 28, 237], [0, 0, 255]]
    for lable_color in lable_color_list:
        GT_red_img[np.where((img == lable_color).all(axis=2))] = [90, 90, 90]
    GT_red_img[np.where((GT_red_img != [90, 90, 90]).all(axis=2))] = blob_color

    # cv2.imshow('test', GT_red_img)
    # cv2.waitKey(0)
    # 3. Fill background color
    background_color = (0, 0, 0)
    rows, cols = GT_blue_img.shape[:2]
    mask = np.zeros((rows + 2, cols + 2), np.uint8)
    retval, GT_blue_img, mask, rect = cv2.floodFill(GT_blue_img, mask, (0, 0), background_color)
    mask = np.zeros((rows + 2, cols + 2), np.uint8)
    retval, GT_red_img, mask, rect = cv2.floodFill(GT_red_img, mask, (0, 0), background_color)

    # cv2.imshow('test', GT_blue_img)
    # cv2.waitKey(0)
    # cv2.imshow('test', GT_red_img)
    # cv2.waitKey(0)

    # 4. blob color > Labeling color로 변환
    GT_blue_img[np.where((GT_blue_img == blob_color).all(axis=2))] = [30, 30, 30]
    GT_red_img[np.where((GT_red_img == blob_color).all(axis=2))] = [90, 90, 90]
    GT_blue_img[np.where((GT_red_img == [90, 90, 90]).all(axis=2))] = [90, 90, 90]

    # cv2.imshow('test', GT_blue_img)
    # cv2.waitKey(0)
    # cv2.imshow('test', GT_red_img)
    # cv2.waitKey(0)
    # cv2.imshow('test', GT_blue_img)
    # cv2.waitKey(0)

    a, b = np.unique(GT_blue_img ,return_counts=True)
    if len(b) == 1:
        return False

    # 5. index값을 가진 마스크 이미지 생성
    label = np.random.randint(3, size=(rows, cols), dtype=np.uint8)
    label.fill(0)
    label[np.where((GT_blue_img == [90, 90, 90]).all(axis=2))] = [1] # red
    label[np.where((GT_blue_img == [30, 30, 30]).all(axis=2))] = [2] # blue
    png = Image.fromarray(label).convert('P')
    cmap = [[0, 0, 0], [255, 0, 0], [0, 0, 255]]
    palette = [value for color in cmap for value in color]
    png.putpalette(palette)
    png.save(dst_full_path)
    return True

def rename(src_name, dst_name):
    os.rename(src_name, dst_name) # 둘 다 절대경로를 필요로 함

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

def repaint(src_name):
    # src_name 파일을 재색칠
    src_color = [255, 127, 36] # 변경 전 색상
    dst_color = [255, 127, 39] # 변경 후 색상

    
if __name__ == '__main__':
    # #test 파일 생성
    # image_full_path, image_name, image_name_without_etx = tools.image_path('D:\\Total_Dataset\\Dataset\\2. Crop Origin Image\\2011028_OK\\wire_barrel_front')
    #
    # for i in range(len(image_full_path)):
    #     tools.convertEtx('tiff', image_full_path[i], image_name_without_etx[i], 'D:\\Total_Dataset\\Dataset\\6. Unlabeld_dataset')

    # crop 이미지 이름 변경
    # dir_path = 'D:\\Total_Dataset\\Dataset\\ori_temp'
    # date_dir_list = ['211102_NG', '211102_OK', '211112_NG', '211112_OK', '2011028_NG', '2011028_OK']
    # part_dir_list = ['sheath']
    # dataset_rename(dir_path, date_dir_list, part_dir_list)

    # """
    # bell mouth 라벨링 이미지 생성
    # src = 3. handmaking Labeling Image
    # dst = 5. segmentation_dataset
    # train, val = 8:2
    # """
    # # 심선 바렐(라벨 이미지가 2개
    # src_base_path = 'D:\\Total_Dataset\\Dataset\\src_temp'
    # dst_base_path = 'D:\\Total_Dataset\\Dataset\\dst_temp'
    # date_dir_list = ['211102_NG', '211102_OK', '211112_NG', '211112_OK', '2011028_NG', '2011028_OK']
    # part_dir_list = ['sheath_barrel', 'sheath_barrel_front', 'bell_mouth','wire_barrel', 'wire_barrel_front']
    # create_mask_dataset(src_base_path, dst_base_path, date_dir_list, part_dir_list)

    """
    original size 라벨 이미지 생성
    """

    src_base_path = 'D:\\src'
    image_full_path_list, image_name_list, image_name_without_etx_list = tools.image_path(src_base_path)
    dst_base_path = 'D:\\temp'
    for image_full_path, image_name_without_etx in zip(image_full_path_list, image_name_without_etx_list):
        original_size_mask(image_full_path, image_name_without_etx, dst_base_path)
