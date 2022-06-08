import numpy as np

# 1차원 배열에서의 np 사용법
# a = np.array([1,2,3,10,20,30,0.1,0.2])
# print(np.min(a)) # 배열에서 최소값을 찾는 함수
# 0.1

# print(np.argmin(a)) # 배열에서 최소값의 위치를 찾는 함수
# 0.6

# print(np.max(a)) # 배열에서 최대값 찾는 함수
# 30.0

# print(np.argmax(a)) # 배열에서 최대값 위치를 찾는 함수
# 5

# print(np.where(a<1)) # 해당 조건에 부합하는 값의 위치를 찾는 함수
#(array([6, 7], dtype=int64),) # 출력은 근본적으로 인덱스형태

# where 응용1
# print(a[np.where(a<1)]) # 해당 조건에 부합하는 값을 반환
# [0.1 0.2]             # 배열 형태를 반환

# where 응용2 : 슬라이싱
# 10보다 크거나 같은 값을 찾아서 0으로 바꾸고 아닌것은 그대로 두라는 조건문을 실행 가능
# print(np.where(a>=10, 0, a))
# [1.  2.  3.  0.  0.  0.  0.1 0.2] # 값을 가진 배열을 반환하기 때문에 a값은 변하지 않음
# print(a) # [ 1.   2.   3.  10.  20.  30.   0.1  0.2]
# a2 = np.where(a>=10, 0, a) # a2라는 변수에 저장
# print(a2)
# [1.  2.  3.  0.  0.  0.  0.1 0.2]

# where 응용3 : 여러개의 조건 설정하기
# a2 = np.where((a >= 10) | (a < 1)) # 10이상 또는 1미만인 요소들만 추출
# print(a2)
# (array([3, 4, 5, 6, 7], dtype=int64),)

# 2차원 배열에서의 np 사용법

# where 응용
# x = np.array([[10, 20, 30],
#                [3, 50, 5]])
# y = np.array([[70, 80, 90],
#              [100, 110, 120]])
# condition = np.where(x>20,x,y) # x > 20의 조건을 만족하면 해당 인덱스의 x값을 넣고, 아니면 y값을 넣음
# [[ 70  80  30] # 위 조건식을 토대로 condition 2차원 배열 생성
#  [100  50 120]]

# 다차원 배열에서의 np 사용법

# squeeze # 다차원 배열에서 차원을 삭제하는 함수
a = np.array([[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]])
# print('a.shape', a.shape)
# a.shape (1, 3, 3)
# print(a)
# [[[1 2 3]
#  [4 5 6]
#  [7 8 9]]]
# print('a.squeeze().shape', a.squeeze().shape)
# a.squeeze().shape (3, 3)
# print(a.squeeze())
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
color = [43,45,200]
print(np.where(
    (color[0] >= 30) & (color[0] <= 50) &
    (color[1] >= 30) & (color[1] <= 50) &
    (color[2] >= 200)
))