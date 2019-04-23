'''
@project : binocular_vision
@author  : Hoodie_Willi
#@description: $将两张图片进行SGBM匹配，得出深度图, 计算深度
#@time   : 2019-04-18 21:39:34
'''
import numpy as np
import cv2
import time
import camera_configs


'''
    创建鼠标点击事件，返回三维坐标
'''
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(threeD[y][x])

'''
    改变参数重新计算视差
'''
def SGBM_update(val=0):
    global threeD
    global SGBM_num
    global SGBM_blockSize
    SGBM_blockSize = cv2.getTrackbarPos('blockSize', 'SGBM_disparity')  # 从滑动条中获取值
    if SGBM_blockSize % 2 == 0:
        SGBM_blockSize += 1
    if SGBM_blockSize < 5:
        SGBM_blockSize = 5
    SGBM_stereo.setBlockSize(SGBM_blockSize)
    SGBM_num = cv2.getTrackbarPos('num_disp', 'SGBM_disparity')
    num_disp = SGBM_num * 16
    SGBM_stereo.setNumDisparities(num_disp)
    SGBM_stereo.setUniquenessRatio(cv2.getTrackbarPos('unique_Ratio', 'SGBM_disparity'))
    SGBM_stereo.setSpeckleWindowSize(cv2.getTrackbarPos('spec_WinSize', 'SGBM_disparity'))
    SGBM_stereo.setSpeckleRange(cv2.getTrackbarPos('spec_Range', 'SGBM_disparity'))
    SGBM_stereo.setDisp12MaxDiff(cv2.getTrackbarPos('disp12MaxDiff', 'SGBM_disparity'))

    print('computing SGBM_disparity...')

    disp = SGBM_stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    threeD = cv2.reprojectImageTo3D(disp, camera_configs.Q)

    # cv2.imshow('left', imgL)
    # cv2.imshow('right', imgR)
    cv2.imshow('SGBM_disparity', (disp - min_disp) / num_disp)
    cv2.imwrite("./detect_img/SGBM_depth.jpg", (disp - min_disp) / num_disp)


if __name__ == "__main__":
    start = time.clock()
    SGBM_blockSize = 5  # 一个匹配块的大小,大于1的奇数
    SGBM_num = 2
    min_disp = 0  # 最小的视差值，通常情况下为0
    num_disp = SGBM_num * 16  # 192 - min_disp #视差范围，即最大视差值和最小视差值之差，必须是16的倍数。
    # blockSize = blockSize #匹配块大小（SADWindowSize），必须是大于等于1的奇数，一般为3~11
    uniquenessRatio = 6  # 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
    speckleRange = 2  # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
    speckleWindowSize = 60  # 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内。
    disp12MaxDiff = 200  # 左右视差图的最大容许差异（超过将被清零），默认为 -1，即不执行左右视差检查。
    P1 = 600  # 惩罚系数，一般：P1=8*通道数*SADWindowSize*SADWindowSize，P2=4*P1
    P2 = 2400  # p1控制视差平滑度，p2值越大，差异越平滑

    imgL = cv2.imread('./detect_img/left.jpg')
    imgR = cv2.imread('./detect_img/right.jpg')

    cv2.namedWindow('SGBM_disparity')
    cv2.setMouseCallback('SGBM_disparity', callbackFunc, None)
    cv2.createTrackbar('blockSize', 'SGBM_disparity', SGBM_blockSize, 21, SGBM_update)
    cv2.createTrackbar('num_disp', 'SGBM_disparity', SGBM_num, 20, SGBM_update)
    cv2.createTrackbar('spec_Range', 'SGBM_disparity', speckleRange, 50, SGBM_update)  # 设置trackbar来调节参数
    cv2.createTrackbar('spec_WinSize', 'SGBM_disparity', speckleWindowSize, 200, SGBM_update)
    cv2.createTrackbar('unique_Ratio', 'SGBM_disparity', uniquenessRatio, 50, SGBM_update)
    cv2.createTrackbar('disp12MaxDiff', 'SGBM_disparity', disp12MaxDiff, 250, SGBM_update)

    SGBM_stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,  # 最小的视差值
        numDisparities=num_disp,  # 视差范围
        blockSize=SGBM_blockSize,  # 匹配块大小（SADWindowSize）
        uniquenessRatio=uniquenessRatio,  # 视差唯一性百分比
        speckleRange=speckleRange,  # 视差变化阈值
        speckleWindowSize=speckleWindowSize,
        disp12MaxDiff=disp12MaxDiff,  # 左右视差图的最大容许差异
        P1=P1,  # 惩罚系数
        P2=P2
    )

    SGBM_update()
    end = time.clock()
    print('SGBM Running time: %s Seconds' % (end - start))
    cv2.waitKey(0)  # 等待键盘输入，如果输入为0则无限等待
    cv2.destroyAllWindows()  # 关闭所有地窗口

