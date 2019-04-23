import cv2
import numpy as np



a = cv2.imread('./left/left01.jpg')
b = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
#图像：输入必须为8位的灰度或彩色图像，尺寸：每幅图像中棋盘的行列数，角落：输出的角点坐标，flag
ret, corners = cv2.findChessboardCorners(a, (7, 6),)
print(ret)
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#img：输入图像，；corner：初始的角点坐标；winsize：大小为窗口的一半
#精细化角点信息,返回每个黑色方块之间的角点在图像坐标系中的位置
corners2 = cv2.cornerSubPix(b, corners, (11, 11), (-1, -1), criteria)
cv2.drawChessboardCorners(a, (7, 6), corners2, True)
cv2.namedWindow('winname', cv2.WINDOW_NORMAL)
cv2.imshow('b', b)
cv2.imshow('winname', a)
cv2.waitKey(0)