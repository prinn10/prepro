import os, cv2
from PIL import Image
# 파일 복사
import shutil
from distutils.dir_util import copy_tree


def image_path(dir_path): # 폴더 경로를 입력받아 폴더 내에 존재하는 이미지의 절대경로, 이미지 이름 list를 반환
    image_full_path_list = [] # absoulute path
    image_name_list = [] # only filename without etx
    image_name_without_etx_list = [] # filename + etx
    for fname in os.listdir(dir_path):
        image_full_path_list.append(os.path.join(dir_path, fname))
        image_name_list.append(fname)
        image_name_without_etx_list.append(fname.rstrip('.bmp'))

    return image_full_path_list, image_name_list, image_name_without_etx_list

def is_image(filename, verbose=False):
    data = open(filename,'rb').read(10)

    # check if file is JPG or JPEG
    if data[:3] == b'\xff\xd8\xff':
        if verbose == True:
             print(filename+" is: JPG/JPEG.")
        return True

    # check if file is PNG
    if data[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
        if verbose == True:
             print(filename+" is: png.")
        return True

    # check if file is GIF
    if data[:6] in [b'\x47\x49\x46\x38\x37\x61', b'\x47\x49\x46\x38\x39\x61']:
        if verbose == True:
             print(filename+" is: gif.")
        return True

    return False

def convertEtx(dst_etx, image_full_path, image_name_without_etx, save_path='default'):
    # 변환 여부 검사
    im = Image.open(image_full_path).convert('RGB')

    #확장자 전처리
    etx1 = dst_etx.lower()
    etx2 = dst_etx.upper()
    if dst_etx == 'tif' or dst_etx == 'tiff':
        etx1 = dst_etx.lower()
        etx2 = 'TIFF'
        im.save(os.path.join(save_path, image_name_without_etx + '.' + etx1), format=str(etx2), compression='tiff_lzw')
        return

    if save_path == 'default':
        im.save(image_name_without_etx + '.' + etx1, etx2)
    else:
        im.save(os.path.join(save_path, image_name_without_etx +'.'+ etx1), etx2)

def remove_file(img_ab_filename, msk_ab_filename = None): # outlier 제거 시 mask, image 함께 제거, unlabel일 경우 image만 제거
    if msk_ab_filename:
        os.remove(msk_ab_filename) # msk_filename 확장자 필요

    os.remove(img_ab_filename) # img_filename 확장자 필요

def create_copy_dataset(src_path, dst_path): # src_path를 dst_path로 복사
    shutil.copytree(src_path, dst_path) # dst_path 존재하지않는 경우
    # copy_tree("./test1", "./test2") # # dst_path 존재하는 경우

def move_matching_file_list(src_folder_path, matching_folder_path, dst_folder_path):
    """
    :param src_folder_path: 해당 경로에 존재하는 파일 목록을 추출
    :param matching_folder_path: src_folder_path에 존재하는 파일 중 src_folder_path에 존재하는 파일과 동일한 파일명을 가진 파일 목록을 추출
    :param dst_folder_path: match_folder_path에서 매칭되는 파일들을 dst_folder_path에 복사하여 저장
    :return: None

    사용 예시: label 이미지가 구성된 폴더를 만들었으나 라벨에 대응되는 원본 이미지를 따로 모아두지 않았을 때
    src_folder_path에 라벨 이미지 폴더경로를 입력하고 macting_folder_path는 원본 이미지를 모두 모아놓은 폴더의 경로를 입력
    dst_folder_path에 src_folder_path의 라벨이미지와 동일한 파일명을 가진 macting_folder_path의 원본 이미지들을 모아서 저장함
    """

    image_full_path_list, image_name_list, _ = image_path(src_folder_path) # src폴더에 존재하는 파일 이름 목록 반환
    matching_image_full_path_list, matching_image_name_list, _ = image_path(matching_folder_path) # matching 폴더에 존재하는 파일 이름 목록 반환

    for matching_image_full_path, matching_image_name in zip(matching_image_full_path_list, matching_image_name_list):
        for image_full_path, image_name in zip(image_full_path_list[::], image_name_list[::]): #[::]없으면 삭제 시 원소 하나 건너뜀
            if image_name == matching_image_name: # mathcing폴더와 src폴더에 동일한 파일이 발견된 경우
                shutil.copyfile(matching_image_full_path, os.path.join(dst_folder_path, matching_image_name))# dst_folder로 파일 이동, matching 폴더에 있는 파일을 옮겨야 함
                image_full_path_list.remove(image_full_path)# image_name_list에서 image_name삭제
                image_name_list.remove(image_name)
                print(matching_image_full_path, '>', os.path.join(dst_folder_path, matching_image_name), len(image_full_path_list))
                break

if __name__ == '__main__':
    move_matching_file_list('Z:\\3. 자체 개발 과제\FineAI\\1. 정희운\Dataset\segmentation_labeling', 'Z:\\3. 자체 개발 과제\\FineAI\\1. 정희운\\Dataset\\original', 'D:\\harness_image')
