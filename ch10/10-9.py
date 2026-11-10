
import cv2 as cv
import mediapipe as mp

# MediaPipe 모듈 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# BlazePose 모델 설정
pose = mp_pose.Pose(
    static_image_mode=False,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 동영상 파일 경로
video_file_path = 'data/purchacing.mp4'
cap = cv.VideoCapture(video_file_path)

# 파일 열기 성공 여부 확인
if not cap.isOpened():
    print(f"오류: 동영상 파일 '{video_file_path}'을 열 수 없습니다.")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()

    # 동영상의 마지막 프레임 확인
    if not ret:
        print('동영상의 마지막 프레임에 도달하여 루프를 종료합니다.')
        break

    # 자세 추정 수행 (BGR -> RGB)
    res = pose.process(
        cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    )

    # 랜드마크 좌표 출력
    if res.pose_landmarks:
        frame_count += 1

        print(f"\n===== Frame {frame_count} =====")

        for idx, landmark in enumerate(res.pose_landmarks.landmark):
            print(
                f"Landmark {idx}: "
                f"x={landmark.x:.4f}, "
                f"y={landmark.y:.4f}, "
                f"z={landmark.z:.4f}, "
                f"visibility={landmark.visibility:.4f}"
            )

    # 랜드마크 그리기
    if res.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            res.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=
                mp_styles.get_default_pose_landmarks_style()
        )

    # 결과 영상 출력
    cv.imshow('MediaPipe pose', frame)

    # q 키를 누르면 종료
    if cv.waitKey(1) == ord('q'):
        # 3차원 랜드마크 시각화
        if res.pose_world_landmarks:
            mp_drawing.plot_landmarks(
                res.pose_world_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
        break

# 자원 해제 및 창 닫기
cap.release()
pose.close()
cv.destroyAllWindows()

# ============================================================
# [실습 10-9] MediaPipe BlazePose를 이용한 자세 추정
#
# 1. 실습 목적
#    - 동영상에서 사람의 신체 자세를 추정한다.
#    - 신체의 주요 관절 위치를 랜드마크로 표시한다.
#    - 각 랜드마크의 좌표 정보를 콘솔에 출력한다.
#
# 2. BlazePose 랜드마크 정보
#    - BlazePose는 총 33개의 신체 랜드마크를 추정한다.
#    - 각 랜드마크는 x, y, z의 3차원 좌표로 표현된다.
#      x: 가로 좌표
#      y: 세로 좌표
#      z: 깊이 좌표
#    - visibility는 해당 랜드마크가 관측될 가능성을
#      나타내는 값이며, 공간 좌표의 차원에는 포함되지 않는다.
#
# 3. 실행 결과
#    - 동영상의 각 프레임에서 사람의 자세를 추정한다.
#    - 추정된 33개 랜드마크와 신체 연결선을 영상에 표시한다.
#    - 각 랜드마크의 x, y, z 및 visibility 정보를 출력한다.
#    - q 키를 누르면 동영상 처리를 종료하고
#      3차원 World Landmark를 시각화한다.
#
# 4. 결론
#    - BlazePose를 이용하여 동영상 속 사람의 자세를
#      추정할 수 있음을 확인하였다.
#    - 랜드마크는 3차원 좌표로 표현되며,
#      visibility를 통해 각 관절의 관측 가능성을 확인할 수 있다.
# ============================================================