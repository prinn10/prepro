import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, cv2
import tools
import tensorflow as tf

# def whereBot(image): #하단기준 찾기
#     img_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     max=-1
#     for y in range (700,1700):
#         for x in range(300,1500):
#             if img_gray[y][x]>max:
#                 max=img_gray[y][x]
#     ret,dst=cv2.threshold(img_gray,max-1,255,cv2.THRESH_BINARY)
#     for y in range (700,1700):
#         b=0
#         for x in range(300,1500):
#             if dst[y][x]!=0:
#                 b=1
#                 break
#     return y

def exam():
    label = np.random.randint(3, size=(10,10), dtype=np.uint8)

    print(label)

    plt.imshow(label, cmap = 'gray')
    plt.show()

    png = Image.fromarray(label).convert('P')
    png.save('label.png')

    palette = [0,0,0,255,0,0,0,128,0]

    png = Image.fromarray(label).convert('P')
    png.putpalette(palette)
    png.save('label.png')

    read = np.asarray(Image.open('basic/label.png'))
    print(read)

import cv2

# 이미지 읽기
img_color = cv2.imread('test1.jpg', cv2.IMREAD_COLOR)

# 컬러 이미지를 그레이스케일로 변환
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 이미지 사이즈 변경
img_gray_reduced = cv2.resize(img_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # 가로, 세로 모두 반으로 줄이기

# 이미지 보여주기
cv2.imshow('color', img_color)  # color라는 이름의 윈도우 안에 img_color라는 이미지를 보여줌
cv2.imshow('gray-scale', img_gray)
cv2.imshow('gray-scale reduced', img_gray_reduced)

cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 저장
cv2.imwrite('result.jpg', img_gray_reduced)  # img_gray_reduced를 result.jpg 이미지 파일로 저장

import cv2

# 이미지 읽기
img = cv2.imread('test1.jpg', cv2.IMREAD_COLOR)

# 사각형 그리기
img = cv2.rectangle(img, (350, 25), (610, 230), (255, 0, 0), 3)
# 사각형을 그릴 이미지, 사각형의 좌측상단좌표, 우측하단좌표, 테두리 색, 테두리 두께
img = cv2.rectangle(img, (490, 180), (760, 360), (255, 0, 0), 3)
img = cv2.rectangle(img, (390, 380), (670, 530), (255, 0, 0), 3)
img = cv2.rectangle(img, (110, 180), (365, 480), (0, 255, 0), 3)

# 텍스트 넣기
cv2.putText(img, 'Coffee', (360, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
# 텍스트를 넣을 이미지, 텍스트 내용, 텍스트 시작 좌측하단좌표, 글자체, 글자크기, 글자색, 글자두께, cv2.LINE_AA(좀 더 예쁘게 해주기 위해)
cv2.putText(img, 'Coffee', (500, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
cv2.putText(img, 'Coffee', (400, 520), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
cv2.putText(img, 'Cake', (120, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2

# 이미지 읽기
img = cv2.imread('test1.jpg', cv2.IMREAD_COLOR)

# 이미지 사이즈 축소
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

cv2.imshow('original image', img)

# 색공간 변환
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR 색공간을 HSV로

# 이미지 채널 분리
(H, S, V) = cv2.split(img)

cv2.imshow('Hue channel', H)
cv2.imshow('Saturation channel', S)
cv2.imshow('Value channel', V)

# Hue 채널값 변화주기
H = H // 2;

# 이미지 채널 병합
HSV = cv2.merge((H, S, V))

# 색공간 변환
img = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

cv2.imshow('new image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
