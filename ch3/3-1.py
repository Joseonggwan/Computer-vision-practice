import cv2 as cv
import sys

img=cv.imread('soccer.jpg')

if img is None:
    sys.exit("Could not read the image.")

cv.imshow('Title', img)
#cv.imshow('Upper left half', img[0:img.shape[0]//2, 0:img.shape[1]//2, :])
#cv.imshow('Center', img[img.shape[0]//4:img.shape[0]*3//4, img.shape[1]//4:img.shape[1]*3//4, :])
cv.imshow('Red channel', img[0:img.shape[0]//2,0:img.shape[1]//2,2])
cv.imshow('Green channel', img[:img.shape[0]//2,0:img.shape[1]//2,1])
cv.imshow('Blue channel', img[:img.shape[0]//2,0:img.shape[1]//2,0])


cv.waitKey()
cv.destroyAllWindows()