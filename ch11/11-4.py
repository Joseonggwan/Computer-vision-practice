from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image

# 1개의 이미지 파일을 열어 PIL Image 객체 리스트로 저장
img=Image.open('data/BSDS_361010.jpg')

# 사전 학습된 DETR 모델 불러오기 (전처리, 가중치)
model_name = 'facebook/detr-resnet-50'
feature_extractor=DetrImageProcessor.from_pretrained(model_name)
model=DetrForObjectDetection.from_pretrained(model_name)

# 이미지 전처리 후 결과 텐서를 torch 형태로 반환
inputs=feature_extractor(images=img, return_tensors='pt')
res=model(**inputs) # softmax 적용 전의 값을 res.logits에 저장함

import numpy as np
import cv2 as cv

colors=np.random.uniform(0,255,size=(100,3))	# 100개 색으로 랜덤하게 트랙 구분
im=cv.cvtColor(np.array(img),cv.COLOR_BGR2RGB) # hugging face to opencv format

pred_boxes = res.pred_boxes[0].cpu().detach().numpy()
print(res.logits.shape) # (batch_size, num_queries, num_classes) = (1, 100, 91)
for i in range(res.logits.shape[1]):
    predicted_label=res.logits[0,i].argmax(-1).item()

    if predicted_label!=91: # DETR은 91개의 클래스 중 예측 (90 : COCO Class + 1 : 'no object')
        name=model.config.id2label[predicted_label]

        prob='{:.2f}'.format(float(res.logits[0,i].softmax(dim=0)[predicted_label]))

        img_width, img_height = img.size
        cx, cy, w, h = pred_boxes[i]
        # print(f"initializing cx, ,cy, w, h: {cx},{cy},{w},{h}")

        # 원본 이미지 위에 BBox와 Label, Prob 표현을 위한 좌표 계산
        cx = int(img_width * cx)
        cy = int(img_height * cy)
        w = int(img_width * w)
        h = int(img_height * h)

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        # 이미지 위에 BBox 및 Label, Prob 출력
        cv.rectangle(im,(x1, y1),(x2, y2),colors[predicted_label],2)
        cv.putText(im,name+str(prob),(x1,y1-5),cv.FONT_HERSHEY_SIMPLEX,0.6,colors[predicted_label],1)

cv.imshow('DETR',im)
cv.waitKey()       
cv.destroyAllWindows()


# [실습 결과]
#
# 1. 두 명의 사람을 person 클래스로 검출하고,
#    두 마리의 말을 horse 클래스로 검출하는 것을 확인하였다.
#
# 2. 사람과 말의 위치에 바운딩 박스가 표시되어
#    이미지 안에서 객체의 종류와 위치를 동시에 파악할 수 있었다.
#
# 3. 일부 객체 주변에 여러 개의 바운딩 박스가
#    겹쳐서 표시되는 현상이 발생하였다.
#    동일한 객체가 중복 검출되는 문제가 있다고 생각한다.
#
# 4. 실제 객체가 있는지 확인하기 어려운 위치에서도
#    sports ball이 검출되는 현상이 나타났다.
#    이는 잘못된 객체 예측일 가능성이 있다.
#
# 5. DETR은 객체의 종류뿐만 아니라 위치까지 예측할 수 있어
#    단순 이미지 분류보다 더 구체적인 정보를 제공한다는 장점이 있다.
#
# 6. 하지만 모든 객체를 정확하게 구분하는 것은 아니며,
#    중복 검출과 오검출을 줄이기 위한 후처리가 필요하다고 생각한다.