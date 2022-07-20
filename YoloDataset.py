import os


import tools


def move_matching_file_list(src_folder_path):
    """
    :param src_folder_path: 해당 경로에 존재하는 파일 목록을 추출
    :param matching_folder_path: src_folder_path에 존재하는 파일 중 src_folder_path에 존재하는 파일과 동일한 파일명을 가진 파일 목록을 추출
    :param dst_folder_path: match_folder_path에서 매칭되는 파일들을 dst_folder_path에 복사하여 저장
    :return: None

    사용 예시: label 이미지가 구성된 폴더를 만들었으나 라벨에 대응되는 원본 이미지를 따로 모아두지 않았을 때
    src_folder_path에 라벨 이미지 폴더경로를 입력하고 macting_folder_path는 원본 이미지를 모두 모아놓은 폴더의 경로를 입력
    dst_folder_path에 src_folder_path의 라벨이미지와 동일한 파일명을 가진 macting_folder_path의 원본 이미지들을 모아서 저장함
    """
    base_path = '/content/drive/MyDrive/Colab Notebooks/YOLOv3/data/plate/images'

    image_full_path_list, image_name_list, image_name_without_etx_list = tools.image_path(src_folder_path) # src폴더에 존재하는 파일 이름 목록 반환
    f = open("traindir.txt", 'w')
    for image_name in image_name_list:
        data = base_path+'/'+image_name
        f.write(data+'\n')
    f.close()


if __name__ == '__main__':
    move_matching_file_list('G:\\내 드라이브\\Colab Notebooks\\YOLOv3\\data\\plate\\images')

