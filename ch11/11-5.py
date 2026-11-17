from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 1개의 이미지 파일을 열어 PIL Image 객체 리스트로 저장
img=Image.open('data/BSDS_361010.jpg')

# 사전 학습된 CLIP 모델 불러오기 (전처리, 가중치)
model_name = 'openai/clip-vit-base-patch32'
processor=CLIPProcessor.from_pretrained(model_name)
model=CLIPModel.from_pretrained(model_name)

# 영상과 비교할 입력 텍스트 4개 정의
captions=['Two horses are running on grass', 'Students are eating', 'Croquet playing on horses', 'Golf playing on horses']
inputs=processor(text=captions,images=img,return_tensors='pt',padding=True)
res=model(**inputs) # res에는 이미지와 텍스트 쌍의 유사도를 계산

import matplotlib.pyplot as plt
plt.imshow(img); plt.xticks([]); plt.yticks([]); plt.show()

# 각 텍스트가 이미지에 적합할 확률 계산 및 출력
logits=res.logits_per_image
probs=logits.softmax(dim=1)
for i in range(len(captions)):
    print(captions[i],': ','{:.2f}'.format(float(probs[0,i].detach() * 100.0)))



# [실습 결과]
#
# CLIP 모델을 이용하여 하나의 이미지와 여러 문장 사이의
# 유사도를 비교하는 실습을 진행하였다.
#
# 이미지와 관련성이 높은 문장은 상대적으로 높은 유사도 값을 보였으며,
# 관련성이 낮은 문장은 낮은 값을 보이는 것을 확인하였다.
#
# 기존 ViT는 이미지의 클래스를 분류하고,
# DETR은 이미지 속 객체의 종류와 위치를 검출하였다.
# 반면 CLIP은 이미지와 텍스트의 의미적 유사도를 비교할 수 있었다.
#
# 이를 통해 CLIP은 미리 정의된 여러 문장 중에서
# 이미지에 가장 적합한 설명을 선택하는 데 활용할 수 있다고 생각한다.