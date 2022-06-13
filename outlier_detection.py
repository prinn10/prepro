import os
import cv2
import numpy as np
import tools
import pandas as pd

import imageinformation
import csv

dst_path = 'D:\\temp_dataset'

# dataset load
dir_path = 'D:\\Total_Dataset\\Dataset\\6. Unlabeld_dataset'
image_paths = [os.path.join(dir_path, image_id) for image_id in sorted(os.listdir(dir_path))]


# 1. rule base outlier detection
def temp_function(image_paths):
    for image_path in image_paths:
        img_info = imageinformation.ImageInformation(image_path)
        # print(img_info.file_name_with_etx)
        means, variances, stds = img_info.printColorInformation()
        if sum(stds) < 30:
            print('아무것도 안찍힘 file name :', img_info.file_name_with_etx)
            tools.remove_file(img_info.absoulute_path)
            print('remove', img_info.file_name_with_etx)
        if sum(variances) > 11600:
            print('소켓이 찍힘', img_info.file_name_with_etx)
            tools.remove_file(img_info.absoulute_path)
            print('remove', img_info.file_name_with_etx)


def temp_function2(image_paths):
    keys = ['file name','mean sum','std sum','var sum']

    name_list = []
    mean_list = []
    var_list = []
    std_list = []

    for image_path in image_paths:
        img_info = imageinformation.ImageInformation(image_path)
        means, variances, stds = img_info.printColorInformation()

        name_list.append(img_info.file_name_with_etx)
        mean_list.append(sum(means))
        var_list.append(sum(variances))
        std_list.append(sum(stds))

    pddd = pd.DataFrame([name_list, mean_list, var_list, std_list], columns=keys)
    print(pddd.head())

# image_paths = [os.path.join(dst_path, image_id) for image_id in sorted(os.listdir(dst_path))]
# temp_function2(image_paths)


def delete_outlier():
    # csv읽어서 outlier 삭제하는 함수
    # 1. 원래 데이터셋 복제
    tools.create_copy_dataset(dir_path, dst_path)

    # 2. read csv
    name_list = []
    f = open('C:\pycharm\source\prepro\data\outlier_list.csv', 'r', encoding='UTF8')
    rdr = csv.reader(f)
    for line in rdr:
        print(line[0])
        name_list.append(line[0])

    for i in range(1, len(name_list)):
        print('delete', os.path.join(dst_path, name_list[i]))
        tools.remove_file(os.path.join(dst_path, name_list[i]))

delete_outlier()


# 2. ML base outlier detection
#
# #  라이브러리
# import os
# import torch
# from torch import nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import MNIST
# from torchvision.utils import save_image
#
# from AE_model import encoder, decoder
# cuda = torch.device('cuda')
# print(cuda)
#
# # 미리 만들어둔 모델 불러오기
#
# def train():
#     #  이미지를 저장할 폴더 생성
#     if not os.path.exists('./AE_img'):
#         os.mkdir('./AE_img')
#
#
#     def to_img(x):
#         x = 0.5 * (x + 1)
#         x = x.clamp(0, 1)
#         x = x.view(x.size(0), 1, 28, 28)
#         return x
#
#
#     img_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#
#     #  Hyper Parameter 설정
#     num_epochs = 100
#     batch_size = 128
#     learning_rate = 1e-3
#
#     #  맨 처음 한번만 다운로드 하기
#     # dataset = MNIST('./data', transform=img_transform, download=True)
#
#     #  데이터 불러오기
#     dataset = MNIST('./data', transform=img_transform)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     #  모델 설정
#     encoder = encoder().cuda().train()
#     decoder = decoder().cuda().train()
#
#     #  모델 Optimizer 설정
#     criterion = nn.MSELoss()
#     encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
#     decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=1e-5)
#
#     for epoch in range(num_epochs):
#         for data in dataloader:
#             img, _ = data  # label 은 가져오지 않는다.
#             img = img.view(img.size(0), -1)
#             img = Variable(img).cuda()
#             # ===================forward=====================
#             latent_z = encoder(img)
#             output = decoder(latent_z)
#             # ===================backward====================
#             loss = criterion(output, img)
#
#             encoder_optimizer.zero_grad()
#             decoder_optimizer.zero_grad()
#             loss.backward()
#             encoder_optimizer.step()
#             decoder_optimizer.step()
#         # ===================log========================
#         print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, float(loss.data)))
#
#         if epoch % 10 == 0:
#             # pic = to_img(output.cpu().data)
#             pic = output.cpu().data
#             pic = pic.view(pic.size(0), 1, 28, 28)
#
#             save_image(pic, './AE_img/output_image_{}.png'.format(epoch))
#
#     #  모델 저장
#     torch.save(encoder.state_dict(), './encoder.pth')
#     torch.save(decoder.state_dict(), './decoder.pth')
#
# def test():
#  pass

