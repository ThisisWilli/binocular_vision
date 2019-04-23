'''
@project : binocular_vision
@author  : Hoodie_Willi
#@description: 利用BM算法得出深度图，算出距离,并拍照
#@time   : 2019-04-09 20:46:36
'''
import numpy as np
import cv2
import camera_configs

cv2.namedWindow("left")
cv2.namedWindow("right")
cv2.namedWindow("depth")
cv2.moveWindow("left", 0, 0)
cv2.moveWindow("right", 600, 0)
cv2.createTrackbar("num", "depth", 0, 10, lambda x: None)
cv2.createTrackbar("blockSize", "depth", 5, 255, lambda x: None)
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 添加点击事件，打印当前点的距离
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(threeD[y][x])


cv2.setMouseCallback("depth", callbackFunc, None)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 从摄像头获取像素
    frame1 = frame[:, 0:640, :]
    frame2 = frame[:, 640:1280, :]
    # 从文件夹中获取图像进行校正，获取深度图
    # frame1 = cv2.imread('./detect_img/left.jpg')
    # frame2 = cv2.imread('./detect_img/right.jpg')
    # print(frame1.shape, frame2.shape)
    # break
    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    # 两个trackbar用来调节不同的参数查看效果
    num = cv2.getTrackbarPos("num", "depth")
    blockSize = cv2.getTrackbarPos("blockSize", "depth")
    if blockSize % 2 == 0:
        blockSize += 1
    if blockSize < 5:
        blockSize = 5

    # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
    stereo = cv2.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)
    disparity = stereo.compute(imgL, imgR)

    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离, 世界坐标系的原点是左摄像头凸透镜的光心
    threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)
    # cv2.filterSpeckles()
    # print(threeD)
    # cv2.imshow("left", imgL)
    # cv2.imshow("right", imgR)
    cv2.imshow("depth", disp)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite('./detect_img/left.jpg', imgL)
        cv2.imwrite('./detect_img/right.jpg', imgR)
        cv2.imwrite('./detect_img/depth.jpg', disp)

cap.release()
cv2.destroyAllWindows()
