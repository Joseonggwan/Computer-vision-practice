import cv2 as cv
import mediapipe as mp

img=cv.imread('data/face.jpg')

mp_face_detection=mp.solutions.face_detection
mp_drawing=mp.solutions.drawing_utils

face_detection=mp_face_detection.FaceDetection(model_selection=1,min_detection_confidence=0.5)
res=face_detection.process(cv.cvtColor(img,cv.COLOR_BGR2RGB))

if not res.detections:
    print('얼굴 검출에 실패했습니다. 다시 시도하세요.')
else:
    for detection in res.detections:
        mp_drawing.draw_detection(img,detection)
    cv.imshow('Face detection by MediaPipe',img)

cv.waitKey()
cv.destroyAllWindows()

#MediaPipe의 BlazeFace 모델을 이용하여 이미지 내 얼굴을 검출하였다. 실행 결과 총 3명의 얼굴이 검출되었으며, 각 얼굴의 위치는 흰색 경계 상자로 표시되었다. 또한 눈, 코, 입 주변의 주요 특징점이 빨간색 점으로 시각화되었다. 이를 통해 BlazeFace가 이미지에서 얼굴의 위치와 주요 특징점을 효과적으로 검출하는 것을 확인하였다.