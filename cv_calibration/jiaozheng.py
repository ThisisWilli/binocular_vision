import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('../data/left*.jpg')
#images += glob.glob('../data/right*.jpg')
for fname in images:
    img = cv2.imread(fname)
    #将图片转化为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print(objp)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners) # 二维坐标点
objpoints = np.array(objpoints)
# print(objpoints, objpoints.shape)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez('B.npz', mtx = mtx, dist = dist, rvecs = rvecs, tvecs = tvecs)
r = np.load('B.npz')
print(r['mtx'], r['dist'], r['rvecs'], r['tvecs'])
a = cv2.imread('left01.jpg')
h, w = a.shape[:2]

#校正图像
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h, w), 1)
dst = cv2.undistort(a, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
#print(roi)

plt.subplot(121), plt.imshow(a), plt.title('source')
plt.subplot(122), plt.imshow(dst), plt.title('undistorted')
plt.show()

