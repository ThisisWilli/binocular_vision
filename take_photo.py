'''
@project : binocular_vision
@author  : Hoodie_Willi
#@description: 左右相机显示并拍照
#@time   : 2019-04-02 14:56:31
'''
import cv2
import numpy as np
import camera_configs

cap = cv2.VideoCapture(1)
# ret = cap.set(3, 320)
# ret = cap.set(4, 240)
# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    left_img = frame[:, 0:640, :]
    right_img = frame[:, 640:1280, :]
    if ret:
        # 显示两幅图片合成的图片
        # cv2.imshow('img', frame)
        # print(frame.shape)
        # 显示左摄像头视图
        cv2.imshow('left', left_img)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        # 显示右摄像头视图
        cv2.imshow('right', right_img)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(delay=2)
    if key == ord('t'):
        cv2.imwrite('./img/left/left' + str(i) + '.jpg', left_img) # 左边摄像头拍照
        cv2.imwrite('./img/right/right' + str(i) + '.jpg', right_img)# 右边摄像头拍照

        # img1_rectified = cv2.remap(left_img, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
        # img2_rectified = cv2.remap(right_img, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
        # imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        # imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
        #
        # cv2.imwrite('./SGBM_test/left' + str(i) + '.jpg', imgL)  # 左边摄像头拍照
        # cv2.imwrite('./SGBM_test/right' + str(i) + '.jpg', imgR)  # 右边摄像头拍照
        i += 1
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
