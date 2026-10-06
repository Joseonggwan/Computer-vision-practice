import cv2
import numpy as np

image = cv2.imread("child.png")

if image is None:
    raise SystemExit("photo.jpg를 실습 폴더에서 찾을 수 없습니다.")

# 화면에 맞게 사진 크기 조절
height, width = image.shape[:2]
scale = min(1.0, 1000 / width, 700 / height)
if scale < 1.0:
    image = cv2.resize(image, None, fx=scale, fy=scale)

height, width = image.shape[:2]
mask = np.full((height, width), cv2.GC_PR_FGD, dtype=np.uint8)

# 바깥쪽 가장자리는 확실한 배경으로 지정
border = max(5, min(width, height) // 40)
mask[:border, :] = cv2.GC_BGD
mask[-border:, :] = cv2.GC_BGD
mask[:, :border] = cv2.GC_BGD
mask[:, -border:] = cv2.GC_BGD

background_model = np.zeros((1, 65), np.float64)
foreground_model = np.zeros((1, 65), np.float64)
has_run = False
drawing = False
draw_label = None
display = image.copy()


def redraw():
    global display
    display = image.copy()

    # 물체 표시(파란색), 배경 표시(빨간색)를 사진 위에 덧씌움
    foreground_strokes = mask == cv2.GC_FGD
    background_strokes = mask == cv2.GC_BGD
    display[foreground_strokes] = (255, 0, 0)
    display[background_strokes] = (0, 0, 255)

    cv2.imshow("GrabCut - 왼쪽: 물체 / 오른쪽: 배경", display)


def mouse_event(event, x, y, flags, param):
    global drawing, draw_label

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        draw_label = cv2.GC_FGD
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        draw_label = cv2.GC_BGD
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        drawing = False
        draw_label = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(mask, (x, y), 8, int(draw_label), -1)
        redraw()


cv2.namedWindow("GrabCut - 왼쪽: 물체 / 오른쪽: 배경")
cv2.setMouseCallback("GrabCut - 왼쪽: 물체 / 오른쪽: 배경", mouse_event)

print("물체는 왼쪽 버튼, 배경은 오른쪽 버튼으로 몇 군데 칠하세요.")
print("g: 분할 실행    r: 표시 초기화    q: 종료")

redraw()

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord("g"):
        cv2.grabCut(
            image,
            mask,
            None,
            background_model,
            foreground_model,
            5,
            cv2.GC_INIT_WITH_MASK
        )
        has_run = True

        foreground = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            1,
            0
        ).astype("uint8")

        result = image * foreground[:, :, np.newaxis]
        cv2.imshow("분할 결과", result)
        redraw()
        print("분할 완료. 더 칠한 뒤 g를 누르면 다시 실행합니다.")

    elif key == ord("r"):
        mask[:] = cv2.GC_PR_FGD
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD
        has_run = False
        redraw()
        print("표시를 초기화했습니다.")

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
