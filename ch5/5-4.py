import cv2 as cv
import numpy as np

# --------------------------------------------------
# 1. 모델 이미지와 장면 이미지 불러오기
# --------------------------------------------------

# 모델 이미지: 찾고 싶은 어린이 보호 표지판
img1 = cv.imread('child.png')

# 장면 이미지: 실제 도로 사진
img2 = cv.imread('road.jpg')

# 이미지가 제대로 불러와졌는지 확인
if img1 is None:
    print('child.png를 불러올 수 없습니다.')
    exit()

if img2 is None:
    print('road.jpg를 불러올 수 없습니다.')
    exit()


# --------------------------------------------------
# 2. 그레이스케일 변환
# --------------------------------------------------

# 모델 이미지를 확대하여 특징점 검출을 쉽게 함
img1 = cv.resize(img1, None, fx=3, fy=3)

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)


# --------------------------------------------------
# 3. SIFT 특징점 검출
# --------------------------------------------------

sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print('모델 이미지 특징점 수:', len(kp1))
print('장면 이미지 특징점 수:', len(kp2))


# --------------------------------------------------
# 4. FLANN을 이용한 특징점 매칭
# --------------------------------------------------

FLANN_INDEX_KDTREE = 1

index_params = dict(
    algorithm=FLANN_INDEX_KDTREE,
    trees=5
)

search_params = dict(
    checks=50
)

flann = cv.FlannBasedMatcher(
    index_params,
    search_params
)

matches = flann.knnMatch(
    des1,
    des2,
    k=2
)


# --------------------------------------------------
# 5. 좋은 매칭 선별
# --------------------------------------------------

good = []

for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good.append(m)

print('전체 매칭 수:', len(matches))
print('좋은 매칭 수:', len(good))


# --------------------------------------------------
# 6. 좋은 매칭이 충분하면 Homography 계산
# --------------------------------------------------

if len(good) >= 4:

    # 모델 이미지의 매칭점
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    # 장면 이미지의 매칭점
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    # RANSAC을 이용하여 Homography 계산
    H, mask = cv.findHomography(
        src_pts,
        dst_pts,
        cv.RANSAC,
        5.0
    )

    print('호모그래피 행렬 H:')
    print(H)


    # --------------------------------------------------
    # 7. 모델 이미지의 네 꼭짓점을 장면 이미지로 변환
    # --------------------------------------------------

    h, w = img1.shape[:2]

    corners = np.float32([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ]).reshape(-1, 1, 2)

    transformed_corners = cv.perspectiveTransform(
        corners,
        H
    )


    # --------------------------------------------------
    # 8. 장면 이미지에 표지판 위치 표시
    # --------------------------------------------------

    result = img2.copy()

    cv.polylines(
        result,
        [np.int32(transformed_corners)],
        True,
        (0, 255, 0),
        3,
        cv.LINE_AA
    )


    # --------------------------------------------------
    # 9. 매칭 결과도 확인
    # --------------------------------------------------

    matchesMask = mask.ravel().tolist()

    matching_result = cv.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        matchesMask=matchesMask,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )


    # --------------------------------------------------
    # 10. 결과 출력
    # --------------------------------------------------

    cv.imshow('Traffic Sign Detection', result)
    cv.imshow('Feature Matching', matching_result)

    cv.waitKey()
    cv.destroyAllWindows()

else:

    print('좋은 매칭점이 4개보다 적어서 Homography를 계산할 수 없습니다.')