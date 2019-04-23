'''
@project : binocular_vision
@author  : Hoodie_Willi
#@description: ${}
#@time   : 2019-04-03 10:42:43
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

path = './img/left2'
# path = './img/right'
objp = np.zeros((6 * 4, 3), np.float32)

#返回多维结构
#将世界坐标系建在标定版上，所有Z坐标都为0，只要赋值x，y
objp[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2) * 10
op = []#所有图片点的三维坐标
imgpoints = []
for i in os.listdir(path):
    file = '/'.join((path, i))
    print(file)
    a = cv2.imread(file)
    b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    #print(b.shape, b.shape[::-1])倒过来
    ret, corners = cv2.findChessboardCorners(a, (6, 4), None)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if ret == True:
        corners2 = cv2.cornerSubPix(b, corners, (11, 11), (-1, -1), criteria)#每幅图片对应的角点数组
        imgpoints.append(corners2)
        op.append(objp)
#mtx为内参矩阵，dist为畸变
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(op, imgpoints, b.shape[::-1], None, None)
print('五个畸变系数为'.format(dist)) #五个畸变参数
print(ret)
print()
print('内参矩阵为\'\n\'{}'.format(mtx))
print()
#外参数
print(len(rvecs), rvecs)
print()
print(len(tvecs), tvecs)
print()
tot_error = 0

#计算误差
for i in range(len(op)):
    #将对象点转换到图像点。然后就可以计算变换得到图像与角点检测算法的绝对差了。然后我们计算所有标定图像的误差平均值。
    imgpoints2, _ = cv2.projectPoints(op[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    tot_error += error

print('平均误差为{}'.format(tot_error / 14))

# a = cv2.imread('left01.jpg')
# h, w = a.shape[:2]
# print(h, w)
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, centerPrincipalPoint=True)
# print()
print(np.random.rand)
