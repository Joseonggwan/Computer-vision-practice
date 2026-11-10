import numpy as np
import cv2 as cv
import sys
import time

def construct_yolo_v3():
    f=open('ch9/coco_names.txt', 'r')
    class_names=[line.strip() for line in f.readlines()]

    model=cv.dnn.readNet('ch9/yolov3-tiny.weights','ch9/yolov3-tiny.cfg')	# 경량 모델 YOLOv3-tiny
    layer_names=model.getLayerNames()
    out_layers=[layer_names[i-1] for i in model.getUnconnectedOutLayers()]

    return model,out_layers,class_names

def yolo_detect(img,yolo_model,out_layers):
    height,width=img.shape[0],img.shape[1]
    test_img=cv.dnn.blobFromImage(img,1.0/256,(416,416),(0,0,0),swapRB=True)

    yolo_model.setInput(test_img)
    output3=yolo_model.forward(out_layers)

    box,conf,id=[],[],[]		# 박스, 신뢰도, 부류 번호
    for output in output3:
        for vec85 in output:
            scores=vec85[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.3:	# 신뢰도가 30% 이상인 경우만 취함 (tiny 모델은 신뢰도가 낮게 나오므로 낮춤)
                centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
                w,h=int(vec85[2]*width),int(vec85[3]*height)
                x,y=int(centerx-w/2),int(centery-h/2)
                box.append([x,y,x+w,y+h])
                conf.append(float(confidence))
                id.append(class_id)

    ind=cv.dnn.NMSBoxes(box,conf,0.3,0.4)
    objects=[box[i]+[conf[i]]+[id[i]] for i in range(len(box)) if i in ind]
    return objects

model,out_layers,class_names=construct_yolo_v3()		# YOLOv3-tiny 모델 생성
colors=np.random.uniform(0,255,size=(len(class_names),3))	# 부류마다 색깔

cap=cv.VideoCapture('ch9/basketball_game.mp4')	# 웹캠 대신 동영상 파일 사용
if not cap.isOpened(): sys.exit('동영상 파일 열기 실패')

start=time.time()
n_frame=0
while True:
    ret,frame=cap.read()
    if not ret: break			# 동영상이 끝나면 루프를 나감

    res=yolo_detect(frame,model,out_layers)

    n_person=0
    for i in range(len(res)):
        x1,y1,x2,y2,confidence,id=res[i]
        if class_names[id]=='person': n_person+=1
        text=str(class_names[id])+'%.3f'%confidence
        cv.rectangle(frame,(x1,y1),(x2,y2),colors[id],2)
        cv.putText(frame,text,(x1,y1+30),cv.FONT_HERSHEY_PLAIN,1.5,colors[id],2)
    cv.putText(frame,'person: '+str(n_person),(10,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)	# 프레임별 사람 수

    cv.imshow("Object detection from video by YOLO v3-tiny",frame)
    n_frame+=1

    key=cv.waitKey(1)
    if key==ord('q'): break

end=time.time()
print('처리한 프레임 수=',n_frame,', 경과 시간=',end-start,'\n초당 프레임 수=',n_frame/(end-start))

cap.release()
cv.destroyAllWindows()

# VideoCapture를 이용하여 동영상 파일을 불러온다.
# cap.read()를 통해 동영상을 프레임 단위로 읽는다.
# 각 프레임에 YOLOv3-tiny를 적용하여 객체를 검출한다.
# 검출된 객체 중 person 클래스의 개수를 계산한다.
# Bounding Box와 Confidence Score를 영상에 표시한다.
# FPS를 계산하여 초당 처리 가능한 프레임 수를 확인한다.