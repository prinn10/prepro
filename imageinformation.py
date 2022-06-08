#이미지 한장을 입력받아서 정보를 출력하는 기능을 제공하는 파일
#출력할 정보들
import cv2
import os
import numpy as np
import tools

class ImageInformation():
    def __init__(self, hint):
        np.set_printoptions(threshold=np.inf, linewidth=np.inf) # 생략없이 출력
        # 파일 정보(경로를 매개변수로 사용)
        self.absoulute_path = os.path.abspath(hint)  # input : path
        self.file_name_with_etx = self.absoulute_path[self.absoulute_path.rfind('\\')+1:]  # input : path, absolute_path, file_name with etx
        self.file_name_without_etx = self.file_name_with_etx[:self.file_name_with_etx.rfind('.')]  # input : file_name with etx
        self.etx = self.file_name_with_etx[self.file_name_with_etx.rfind('.')+1:]   # input : path, absolute_path, file_name with etx
        # self.etx_integrity = tools.is_image(self.absoulute_path)  # type: boolean # input: absolute_path

        # 이미지 정보(이미지를 매개변수로 사용)
        self.image = cv2.cvtColor(cv2.imread(self.absoulute_path), cv2.COLOR_BGR2RGB)
        self.image_size = self.image.size  # input : image , image size (byte)
        self.image_dtype = self.image.dtype  # input : image , image type : uint8 or float16 etc....
        self.image_color_R, self.image_color_G, self.image_color_B = cv2.split(self.image)  # input image
        self.image_width, self.image_height, self.image_channel = self.image.shape  # input : image
        self.image_type = type(self.image)  # input : image, type : class
        # self.image_color_type # RGB, BGR, GrayScale

        # 마스크일 경우 해당되는 정보들
        self.image_index = np.array(self.image_color_R)  # input : image.split color , same R, G, B, if image is label, index vaild # 라벨링 마스크 값
        self.image_class = np.unique(self.image_index)  # input : iamge_index , type : list # 클래스 종류
        self.image_class_num = len(np.unique(self.image_index))  # input : image_index, type : int # 클래스의 개수

    def printMetaInformation(self):
        # 확장자
        print('확장자', self.etx)
        # 확장자의 무결성 검사 결과
        # print('확장자의 무결성 검사 결과', self.etx_integrity)
        # 파일 경로(절대경로, 파일이름, 확장자가 없는 파일이름)
        print('파일 경로(절대경로)', self.absoulute_path)
        print('파일 경로(파일 이름)', self.file_name_with_etx)
        print('파일 경로(확장자가 없는 파일 이름)', self.file_name_without_etx)

        # 이미지 정보
        print('이미지 높이, 크기, 채널', self.image_width, self.image_height, self.image_channel)
        print('이미지 크기', self.image_size)
        print('이미지 타입', self.image_dtype)
        print('이미지 타입2', self.image_type)

    def showImage(self):
        # 각 픽셀에 대한 값 ( R,G,B, index값, unique 값)
        print('각 픽셀에 대한 값 Red')
        print(self.image_color_R)
        print('각 픽셀에 대한 값 Green')
        print(self.image_color_G)
        print('각 픽셀에 대한 값 Blue')
        print(self.image_color_B)
        cv2.imshow(self.file_name_with_etx, self.image)
        cv2.waitKey(0)

    def showIndex(self):
        # print('마스크 인덱스')
        # print(self.image_index)
        print('마스크 클래스 종류', self.image_class)
        print('마스크 클래스 종류 개수', self.image_class_num)
        unique, counts = np.unique(self.image, return_counts=True)
        print('각 클래스 별 개수', dict(zip(unique, counts)))

        # 마스크일 경우 해당되는 정보들

    def setFileName(self, file_name, etx='.png'):
        # 파일명에 확장자가 포함된 경우
        if file_name.find('.') != -1:
            self.file_name_with_etx = file_name
            self.file_name_without_etx = self.file_name_with_etx[:self.file_name_with_etx.rfind('.')]

        # 파일명에 확장자가 포함되지 않은 경우
        else:
            self.file_name_with_etx = file_name + etx
            self.file_name_without_etx = file_name

    def saveImage(self, save_full_path_with_etx):
        cv2.imwrite(save_full_path_with_etx, self.image)
