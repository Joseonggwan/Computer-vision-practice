import cv2 as cv

# 이미지 불러오기
img = cv.imread('rose.png')

# 마우스로 선택할 영역의 좌표
x1, y1 = -1, -1
x2, y2 = -1, -1
selecting = False


# 마우스 이벤트 함수
def mouse_callback(event, x, y, flags, param):
    global x1, y1, x2, y2, selecting

    # 마우스 왼쪽 버튼을 누르면 시작점 저장
    if event == cv.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        selecting = True

    # 마우스를 움직이는 동안 선택 영역 표시
    elif event == cv.EVENT_MOUSEMOVE and selecting:
        temp = img.copy()
        cv.rectangle(temp, (x1, y1), (x, y),
                     (255, 0, 0), 3)
        cv.imshow('Select ROI', temp)

    # 마우스 왼쪽 버튼을 놓으면 끝점 저장
    elif event == cv.EVENT_LBUTTONUP:
        x2, y2 = x, y
        selecting = False

        temp = img.copy()
        cv.rectangle(temp, (x1, y1), (x2, y2),
                     (255, 0, 0), 3)
        cv.imshow('Select ROI', temp)


# 선택 창 생성
cv.namedWindow('Select ROI')
cv.setMouseCallback('Select ROI', mouse_callback)

# 원본 이미지 표시
cv.imshow('Select ROI', img)

# 영역 선택 후 키 입력
cv.waitKey()


# 선택 영역의 좌표 정리
x = min(x1, x2)
y = min(y1, y2)
w = abs(x2 - x1)
h = abs(y2 - y1)


# 선택한 영역 추출
patch = img[y:y+h, x:x+w, :]


# 선택 영역 표시
original = img.copy()
cv.rectangle(original, (x, y), (x+w, y+h),
             (255, 0, 0), 3)


# 최근접 보간
patch1 = cv.resize(
    patch,
    dsize=(0, 0),
    fx=5,
    fy=5,
    interpolation=cv.INTER_NEAREST
)


# 양선형 보간
patch2 = cv.resize(
    patch,
    dsize=(0, 0),
    fx=5,
    fy=5,
    interpolation=cv.INTER_LINEAR
)


# 양3차 보간
patch3 = cv.resize(
    patch,
    dsize=(0, 0),
    fx=5,
    fy=5,
    interpolation=cv.INTER_CUBIC
)


# 결과 출력
cv.imshow('Original', original)
cv.imshow('Resize nearest', patch1)
cv.imshow('Resize bilinear', patch2)
cv.imshow('Resize bicubic', patch3)

cv.waitKey()
cv.destroyAllWindows()