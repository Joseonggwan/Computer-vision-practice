import cv2 as cv 
import numpy as np

img = cv.imread('soccer.jpg')  # 영상 읽기
if img is None:  # 이미지가 제대로 로드되지 않은 경우
    raise FileNotFoundError("Image file 'soccer.jpg' not found. Please check the file path.")

img_show = np.copy(img)  # 붓 칠을 디스플레이할 목적의 영상

mask=np.zeros((img.shape[0],img.shape[1]),np.uint8) 
mask[:,:]=cv.GC_PR_BGD		# 모든 화소를 배경일 것 같음으로 초기화

BrushSiz=9				# 붓의 크기ㅂ
LColor,RColor=(255,0,0),(0,0,255)	# 파란색(물체)과 빨간색(배경)

def painting(event,x,y,flags,param):
    if event==cv.EVENT_LBUTTONDOWN:   
        cv.circle(img_show,(x,y),BrushSiz,LColor,-1)	# 왼쪽 버튼 클릭하면 파란색
        cv.circle(mask,(x,y),BrushSiz,cv.GC_FGD,-1)
    elif event==cv.EVENT_RBUTTONDOWN: 
        cv.circle(img_show,(x,y),BrushSiz,RColor,-1)	# 오른쪽 버튼 클릭하면 빨간색
        cv.circle(mask,(x,y),BrushSiz,cv.GC_BGD,-1)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        cv.circle(img_show,(x,y),BrushSiz,LColor,-1)# 왼쪽 버튼 클릭하고 이동하면 파란색
        cv.circle(mask,(x,y),BrushSiz,cv.GC_FGD,-1)
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_RBUTTON:
        cv.circle(img_show,(x,y),BrushSiz,RColor,-1)	# 오른쪽 버튼 클릭하고 이동하면 빨간색
        cv.circle(mask,(x,y),BrushSiz,cv.GC_BGD,-1)

    cv.imshow('Painting',img_show)
    
cv.namedWindow('Painting')
cv.imshow('Painting', img_show)  # 초기 이미지를 먼저 디스플레이
cv.setMouseCallback('Painting',painting)

while(True):				# 붓 칠을 끝내려면 'q' 키를 누름
    if cv.waitKey(1)==ord('q'): 
        break

# 여기부터 GrabCut 적용하는 코드
background=np.zeros((1,65),np.float64)	# 배경 히스토그램 0으로 초기화
foreground=np.zeros((1,65),np.float64)	# 물체 히스토그램 0으로 초기화q

cv.grabCut(img,mask,None,background,foreground,1,cv.GC_INIT_WITH_MASK)
mask2=np.where((mask==cv.GC_BGD)|(mask==cv.GC_PR_BGD),0,1).astype('uint8')
grab=img*mask2[:,:,np.newaxis]
cv.imshow('Grab cut image',grab)  

cv.waitKey()
cv.destroyAllWindows()


# cv.grabCut(img, mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
# 5는 GrabCut 알고리즘의 반복 횟수를 의미한다.
# 전경과 배경을 반복적으로 추정하여 영상 분할 결과를 개선한다.

# cv.grabCut(img, mask, None, background, foreground, 1, cv.GC_INIT_WITH_MASK)
# 반복 횟수를 5회에서 1회로 줄였다.
# 반복 횟수가 줄어들어 전경과 배경을 추정하고 분할 결과를 개선할 기회가 감소한다.
# 따라서 5회 반복한 경우보다 객체와 배경의 분리가 덜 정교하게 나타날 수 있다.