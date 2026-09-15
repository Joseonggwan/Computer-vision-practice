import cv2 as cv

img = cv.imread('rose.png')

# 마우스로 영역 선택
x, y, w, h = cv.selectROI('Select ROI', img, fromCenter=False, showCrosshair=True)

# 선택한 영역 추출
patch = img[y:y+h, x:x+w, :]

# 선택 영역 표시
img = cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

# 보간법에 따른 확대
patch1 = cv.resize(patch, dsize=(0, 0), fx=5, fy=5,
                   interpolation=cv.INTER_NEAREST)

patch2 = cv.resize(patch, dsize=(0, 0), fx=5, fy=5,
                   interpolation=cv.INTER_LINEAR)

patch3 = cv.resize(patch, dsize=(0, 0), fx=5, fy=5,
                   interpolation=cv.INTER_CUBIC)

cv.imshow('Original', img)
cv.imshow('Resize nearest', patch1)
cv.imshow('Resize bilinear', patch2)
cv.imshow('Resize bicubic', patch3)

cv.waitKey()
cv.destroyAllWindows()