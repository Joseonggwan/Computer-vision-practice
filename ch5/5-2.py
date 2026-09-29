import cv2 as cv

img = cv.imread('mot_color70.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

kp, des = sift.detectAndCompute(gray, None)

print('특징점 개수:', len(kp))

img = cv.drawKeypoints(
    gray,
    kp,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv.imshow('SIFT', img)
cv.waitKey()
cv.destroyAllWindows()