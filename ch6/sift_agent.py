import cv2

# 비교 기준이 될 표지판 이미지
sign_files = ["child.png", "elder.png", "disabled.png"]

# 이번에 인식해 볼 이미지
target_file = "elder.png"

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_L2)

target = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
if target is None:
    raise SystemExit(f"{target_file} 파일을 읽지 못했습니다.")

target_keypoints, target_descriptors = sift.detectAndCompute(target, None)
if target_descriptors is None:
    raise SystemExit("대상 이미지에서 특징점을 찾지 못했습니다.")

best_name = None
best_image = None
best_keypoints = None
best_matches = []
all_results = []

for filename in sign_files:
    sign = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if sign is None:
        print(f"{filename} 파일을 읽지 못했습니다.")
        continue

    sign_keypoints, sign_descriptors = sift.detectAndCompute(sign, None)
    if sign_descriptors is None:
        print(f"{filename}: 특징점을 찾지 못했습니다.")
        continue

    pairs = matcher.knnMatch(
        target_descriptors, sign_descriptors, k=2
    )

    good_matches = []
    for pair in pairs:
        if len(pair) == 2:
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

    all_results.append((filename, len(good_matches)))

    if len(good_matches) > len(best_matches):
        best_name = filename
        best_image = sign
        best_keypoints = sign_keypoints
        best_matches = good_matches

print("표지판별 특징점 일치 수:")
for filename, count in all_results:
    print(f"  {filename}: {count}")

if best_name is None:
    raise SystemExit("비교할 표지판 이미지를 찾지 못했습니다.")

print(f"\n가장 비슷한 표지판: {best_name}")

match_view = cv2.drawMatches(
    target,
    target_keypoints,
    best_image,
    best_keypoints,
    best_matches[:40],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imshow("SIFT 표지판 비교", match_view)
print("비교 창을 닫으려면 q 키를 누르세요.")

while True:
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
