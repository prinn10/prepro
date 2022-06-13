import os, cv2
from PIL import Image
# 파일 복사
import shutil
from distutils.dir_util import copy_tree


def image_path(dir_path): # 폴더 경로를 입력받아 폴더 내에 존재하는 이미지의 절대경로, 이미지 이름 list를 반환
    image_full_path = [] # absoulute path
    image_name_without_etx = [] # filename + etx
    image_name = [] # only filename without etx
    for fname in os.listdir(dir_path):
        image_full_path.append(os.path.join(dir_path, fname))
        image_name.append(fname)
        image_name_without_etx.append(fname.rstrip('.bmp'))

    return image_full_path, image_name, image_name_without_etx

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

if __name__ == '__main__':
    image_path()

