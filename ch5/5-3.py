import cv2 as cv
import numpy as np

img1 = cv.imread('mot_color70.jpg')
img2 = cv.imread('mot_color83.jpg')

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

print('전체 매칭 수:', len(matches))
print('좋은 매칭 수:', len(good))

result = cv.drawMatches(
    img1, kp1,
    img2, kp2,
    good, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv.imshow('FLANN Matching', result)
cv.waitKey()
cv.destroyAllWindows()