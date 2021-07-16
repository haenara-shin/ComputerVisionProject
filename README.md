# Deep Learning Computer Vision

```bash
|-- Introduction of Object Detection
|  |-- Understanding of Object detection & Segmentation
|  |-- Main properties of Object detection and difficulties
|  |-- Understanding of Object Localization & Detection
|     |-- Region Proposal
|     |-- IoU
|     |-- NMS
|     |-- mAP
|-- Introduction of OpenCV and Dataset for Object detection & Segmentation
|-- RCNN Object Detecter (RCNN, SPPNet, Fast RCNN, Faster RCNN)
|  |-- RCNN
|  |-- SPPNet
|  |-- Fast RCNN
|  |-- Faster RCNN
|-- Understanding of MMDetection & Faster RCNN
|-- SSD
|-- YOLO
|-- Ultralytics Yolo
|-- RetinaNet & EfficientDet
|-- AutoML EfficientDet
|-- Segmentation - Mask RCNN
```

# Introduction of Object Detection 
## Object detection 과 Segmentation 이해
- AlexNet (2014)의 등장: Imagenet 압도적 성능 개선 -> Deep learning 기반으로 발전
	- PASCAL VOC (object detection 대회) 에서 mAP (mean Average Precision) 성능 대폭 개선하기 시작(약 60% 이상) (그전 모델인 LeNet 부터 큰폭 성능 개선 시작함. Deep ConvNets 사용) 
- Classification: 하나의 image/object에 대해서 분류
- Localization: `단 하나의 object` 위치를 bounding box로 지정하여 찾음.
- Detection: `여러 개의 objects` 에 대한 위치를 `Bounding box`로 지정하여 찾음.
- Segmentation: `여러 개의 objects` 에 대한 위치를 bounding box가 아니라 `픽셀 단위`로 분류.

- Localization/Detection 은 해당 object의 `위치`를 `bounding box`로 찾고, `bounding box`내의 object를 판별함.
	- Bounding box regression(box의 좌표값을 예측)과 classification 두 개의 문제가 합쳐져 있음.
	- Localization에 비해 Detection은 두 개 이상의 objects 를 하나의 이미지의 임의 위치에서 찾아야 하므로 상대적으로 localization 보다 여러가지 어려운 문제에 봉착하게 됨.
- `One-stage detector`: 바로 detect 적용. (`YOLO`, `SSD`, `Retina-Net`, `EfficientDet`)
- `Two-stage detector`: 오브젝트가 특정 위치에 있을 법한 곳을 예측해서 찾아보고, 그리고 detect. (`RCNN` 계열) inference가 느림(실시간 적용이 어렵다.)
![DLCV_history](https://user-images.githubusercontent.com/58493928/125342910-4f077900-e30a-11eb-86e5-c4d1f3fe9fc8.png)

## Object detection 주요 구성 요소 및 왜 object detection 은 어려운가?
### Object detection 주요 구성 요소
1. 영역 추정: Region proposal
   1. bounding box 의 위치를 추정(regression)
   2. bounding box 내부의 object를 탐지(classification)
   3. 영역 추정: object 가 있을 만한 위치를 추정
2. Detection을 위한 DL 네트웍 구성
   1. Backbone(BB): Feature extraction (Image classification)
   2. Neck: FPN (Feature Pyramid Network) - BB 와 Head 연결
   3. Head: Network prediction (classification + regression)
3. Detection 을 구성하는 기타 요소
   1. `IoU` (Intersection of Union)
   2. `NMS` (Non Max Suppression)
   3. `mAP` (mean Average Precision)
   4. `Anchor box`
- Backbone (feature extractor, ResNet) + Neck (Feature Pyramid Network, FE에서 작은 오브젝트를 인식하기 위해 도입) + Head (classification + bounding box(bb) regression)
### Object detection 어려운 이유
1. Classification + Regression 동시에 해야함: 이미지에서 여러 개의 물체를 Classification 함과 동시에 위치를 찾아야(regression) 함.
   1. classification/regression 각 loss 함수에 대한 설정/최적화가 쉽지 않음.
2. 다양한 크기와 유형의 object가 섞여 있음: 크기가 서로 다르고, 생김새가 다양한 오브젝트가 섞여 있는 이미지에서 이들을 detect 해야함.
3. Detect 시간: 실시간 영상 기반의 경우, detect 시간이 중요함.
4. 명확하지 않은 이미지: 오브젝트 이미지가 명확하지 않거나, 전체 이미지에서 detect 할 오브젝트가 차지하는 비중이 높지 않은 경우도 있음(배경이 대부분을 차지 한다든지.)
5. 데이터 세트 부족: 훈련 가능한 데이터 세트가 부족하며 annotation 을 만들어야 함(4번 이유의 연장선)

## Object Localization 과 Detection 이해

  <img width="1122" alt="스크린샷 2021-07-14 오전 11 10 47" src="https://user-images.githubusercontent.com/58493928/125671574-201a7af6-70a3-4b4e-8b1b-243eab81226a.png">

  <img width="1138" alt="스크린샷 2021-07-14 오전 11 14 54" src="https://user-images.githubusercontent.com/58493928/125672098-a65122a2-fcff-4be6-aaf9-06745f69c7a8.png">

### Object Localization 개요
- 원본 이미지에 하나의 오브젝트만 있는 경우
- Image classification + (Annotation file에 위치 정보 표현 갖고 있음 --> feature map 에 오브젝트가 하나 밖에 없어서 Bounding Box Regression --> 학습할 수록 bounding box 위치 좌표(annotation) 예측 오류를 줄여나감.)
- 하지만 이런 로직은 object detection 즉 두 개 이상의 오브젝트를 검출해야 할 때 문제가 생김. 이미지의 어느 위치에서 오브젝트를 찾아야 하는지?? --> 위치 기반으로 찾으면 엉뚱한 위치를 찾아감. --> inference 가 어렵다. 
- `있을 만한 위치(region proposal, 영역 추정)`이 중요함
  
### Object Detection (`region proposal, 영역 추정`)
1. 먼저, Sliding Windows 방식
   1. 윈도우를 왼쪽 상단에서 부터 오른쪽 하단으로 이동시키면서 오브젝트를 detection 하는 방식 (초기 기법)
   2. 오브젝트 없는 영역도 무조건 슬라이딩 해야 하며 `여러 형태의 윈도우` or/and `여러 스케일을 가진 이미지`를 스캔해서 검출해야 하므로 수행 시간이 오래 걸리고 검출 성능이 상대적으로 낮음.
2. Region proposal (영역 추정) 방식 (`Selective Search`)
   1. `오브젝트가 있을 만한 후보 영역을 찾자`
   2. 원본이미지에 대해서 후보 bounding box 선택
      1. pixel intensity 기반한 graph-based segment 기법에 따라 over segmentation 을 수행함 --> 각각의 오브젝트들이 1개의 개별 영역에 담길 수 있도록 많은 초기 영역을 생성 
      2. 각 segmentation 을 각각의 bounding box 로 만든다. --> region proposal 리스트로 추가
      3. 컬러, 무늬(texture), 크기(size), 형태(shape)에 따라서 유사한 Region을 계층적 그룹핑 방법으로 계산하고 segment 들을 그룹핑함.
      4. 1-3 과정을 반복함.
   3. 최종 후보 도출해서 최종 오브젝트 detection

### Object Detection Metric - IoU
- Intersection over Union
- 모델이 예측한 결과와 실측(ground truth) box가 얼마나 정확하게 겹치는가를 나타냄.
- IoU = (Area of Overlap) / (Area of Union)

### Object Detection - NMS
- Non Max Suppression
- Detected 오브젝트의 bounding box 중에 비슷한 위치에 있는 box를 제거하고(눌러준다? 억제한다?) 가장 적합한 box를 선택하는 기법
- 수행 로직
  1. Detected 된 bounding box 별 특정 confidence threshold 이하 bounding box는 먼저 제거(confidence score < 0.5)
  2. 가장 높은 confidence score를 가진 box 순으로 내림차순 정렬하고 아래 `로직`을 모든 box에 순차적 적용.
     1. `로직`: 높은 confidence score를 가진 box와 겹치는 다른 box를 모두 조사해서 IoU가 특정 threshold 이상인 box를 모두 제거(IoU threshold > 0.4) --> 높은 confidence score 가지는 박스와 겹치는 박스를 제거
  3. 남아 있는 box만 선택
- `Confidence score`가 `높을 수록`, `IoU threshold`가 `낮을 수록` 많은 box가 제거됨.

### Object Detection - mAP
- mean Average Precision
- 실제 오브젝트가 detected 된 재현율(recall)의 변화에 따른 정밀도(precision)의 값을 평균한 성능 수치
  - 여기서 잠깐!!!! <`정밀도(precision)`과 `재현율(recall)`>
  - ![스크린샷 2021-07-13 오후 4 49 21](https://user-images.githubusercontent.com/58493928/125539489-2edf52d7-17e0-4a34-8b3d-31f2e24bf62e.png)
    - 1. 정밀도
      - `예측을 positive로 한 대상 중` 예측과 실제 값이 positive로 일치한 데이터의 비율.
      - `실제 negative`인 데이터 예측을 positive로 잘못 판단하면 큰일 나는 경우. (ex: 스팸메일) --> `FP` (Type I error)
      - detection 결과가 실제 object 들과 얼마나 일치하는지.
    - 2. 재현율
      - `실제 값이 positive한 대상 중`에서 예측과 실제 값이 positive로 일치한 데이터의 비율.
      - `실제 positive` 데이터 예측을 negative로 잘못 판단하면 큰일 나는 경우. (ex: cancer, finance fraud) --> `FN` (Type II error)
      - detection 알고리즘이 실제 object들을 빠뜨리지 않고 얼마나 정확히 검출 예측 하는지를 나타냄.
  - ![스크린샷 2021-07-13 오후 4 33 31](https://user-images.githubusercontent.com/58493928/125538480-8d71fcaa-3582-4050-baf4-b3431745f609.png)
- 검출 예측이 성공했는지 여부를 IoU 로 결정하는데, 
  - PASCAL VOC 에서 사용된 기준은 0.5 이상이면 True Positive, COCO 에서는 다른 기준 적용함.
  - COCO Challenge 에서의 mAP는 IoU를 다양한 범위로 설정하여 예측 성공 기준을 정함.
    - IoU 0.5 부터 0.05씩 값을 증가 시켜서 0.95 까지 해당하는 IoU 별로 mAP를 계산(`AP@[.50:.05:.95]`)
- ![스크린샷 2021-07-13 오후 4 38 50](https://user-images.githubusercontent.com/58493928/125538721-6dad6bd1-3439-40d4-99fe-383cde05fa29.png)
- Confidence threshold 값이 낮을 수록 더 많은 예측 bounding box를 만들게 되어 정밀도는 낮아지고 재현율은 높아짐. (반대로 값이 높을 수록 정밀도는 높아지고 재현율은 낮아짐.)
- 정밀도-재현율 곡선(precision-recall curve)
  - recall 값의 변화에 따른 (confidence threshold 값을 조정하면서 얻어진) precision 값을 나타낸 곡선을 정밀도-재현율 곡선이라고 함. 
  - 얻어진 precision 값의 평균을 AP 라고 하며, 일반적으로 곡선의 면적 값으로 계산됨.
  - ![스크린샷 2021-07-13 오후 5 06 01](https://user-images.githubusercontent.com/58493928/125540594-04c2cf98-f466-4c51-a78b-3191cc6451a4.png)
  - 오른쪽 최대 precision  값을 연결한다 --> 너비가 average precision (AP) 의미.
    - AP는 한 개의 오브젝트에 대한 의미. mAP 는 그런 오브젝트들의 AP 평균

# Introduction of OpenCV and (major) Dataset for Object detection & Segmentation
## Dataset
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/): `xml format`. 20개의 오브젝트 카테고리. 개별 오브젝트의 bounding box 정보(xmin, ymin, xmax, ymax)
- [MS COCO](http://cocodataset.org/#download): `json format`. 80개의 오브젝트 카테고리(총 91개 ID 중 데이터셋이 없는 ID가 11개 있음.). 많은 오픈 소스 계열의 주요 패키지들의 pretrained model 바탕이 되는 데이터셋. bounding box 정보가 소숫점으로 표시되기 때문에 정수값으로 바꿔야 함. 이미지 한 개에 여러 오브젝트들을 가지고 있기 때문에 타 데이터셋에 비해 난이도가 높은 데이터 제공.
- Google Open Images: `csv format`. 600개의 오브젝트 카테고리
## OpenCV
- `PIL`: 처리 성능이 상대적으로 느림. `Image.oepn()`으로 image file을 읽어서 ImageFile객체로 생성.
- `scikit-image`: scipy, numpy 기반
- `opencv`: c++ 기반
  -  사용상 주의점 (1)
     - `imread('파일명')` 은 파일을 읽어 numpy array 로 변환하는데, BGR 형태로 로딩하기 때문에 `cvtColor(imread로 읽어와서 저장한 변수명, cv2.COLOR_BGR2RGB)` 처리 해야함.
      ```python
      import cv2
      import matplotlib.pyplot as plt

      img_bgr = cv2.imread('파일명')
      img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
      plt.imshow(img_rgb)
      ```
  - 사용상 주의점(2)
    - `imwrite`를 사용하면 다시 RGB로 저장됨. 굳이 RGB로 변환하는 절차를 거칠 필요 없음.
  - 사용상 주의점(3)
    - `cv2.imshow(이미지array)`는 이미지 배열을 window frame에 보여줌. 근데 주피터노트북에서는 에러가 발생. `cv2.waitKey()`는 키보드 입력이 있을 때까지 무한 대기. `cv2.destroyAllWindows()` 화면의 윈도우 프레임 모두 종료. 그래서 `이미지 배열 시각화 할때 주피터노트북에서는 matplotlib` 사용. `plt.imshow(배열, 이미지, 다 읽을 수 있음)`
  - `cv2.VideoCapture(처리할 파일)`는 동영상을 개별 frame 으로 하나씩 읽어들임(`.read()`). `cv2.VideoWriter(출력될 파일명)`는 `VideoCapture`로 읽어들인 개별 frame 을 `동영상 파일로 Write` 수행.

#  RCNN Object Detecter (RCNN, SPPNet, Fast RCNN, Faster RCNN)
- Object localization: 이미지 1개에 1개의 오브젝트
- Object detection: 이미지 1개에 2개 이상의 오브젝트
  - Object가 있을 만한 위치를 추정: Sliding window 방식(학습 시간, window/이미지 크기/스케일에 영향) / Region Proposal 방식(Selective Search - Segmentation)
## RCNN (Region Proposal 기반의 Object detection model)
- Region Proposal 방식 도입.
- Stage1: Region proposal (object가 있을 만한 위치를 proposal/뽑음 (약 2000개) --> 각 영역별 이미지(를 image crop과 wrap 적용해서 만듦)의 사이즈를 동일하게 맞춘 뒤(CNN detecter에서 FC layer에 넣을때 이미지 크기가 동일해야함.))
- Stage2: 앞서의 이미지를 CNN detecter 구조에 넣음.
- 근데, CNN detecter 의 마지막 레이어에서 최종 FC layer 결과를 `SVM classifier`에 넣음. (해봤더니 좋아서 했음...) 또한 Bbox reg에 넣음.
- 각 이미지를 ConvNet 거치는데, 그런걸 2000개 씩... 
<img width="1146" alt="스크린샷 2021-07-16 오전 10 47 07" src="https://user-images.githubusercontent.com/58493928/125988471-77679fe7-d1ea-40bc-9636-79f346ebd8d3.png">
<img width="1169" alt="스크린샷 2021-07-16 오전 11 04 38" src="https://user-images.githubusercontent.com/58493928/125990433-c236f598-0dd6-4218-ae97-9b8f56125814.png">
<img width="1071" alt="스크린샷 2021-07-16 오전 11 05 04" src="https://user-images.githubusercontent.com/58493928/125990469-9a1dbc3e-6c29-4cda-80cc-5264f9530483.png">
  - 학습을 통해 dx(p), dy(p)를 구함.
- 동시대의 다른 알고리즘 대비 높은 detection 정확도. 하지만, 너무 느린 detection 시간과 복잡한 아키텍쳐 및 학습 프로세스
  - 하나의 이미지 마다 selective search를 수행해서 약 2000개의 region 영역 이미지들 도출.
  - 개별 이미지 별로 2000개씩 생성된 region 이미지에 대한 CNN feature map 생성.
  - 1장의 이미지에 대한 object detection 하는데 약 50초 소요.

## SPPNet (Spatial Pyramid Pooling Net)
- 2000개의 Region Proposal 이미지를 Feature extraction 하지 않고, 원본 이미지만 CNN 으로 Feature map 생성한 뒤에, 원본 이미지의 selective search로 추천된 영역의 이미지만 feature map 으로 mapping 해서 별도 추출하기 위해서(예전에는 서로 다른 사이즈의 image를 CNN에 넣으면 FC layer의 크기가 고정 되어야 하는 문제 발생했었으니까), 서로 다른 크기를 가진 region proposal 이미지를 `SPP Net의 고정된 크기 vector로 변환한 뒤에` 1D Flatten FC layer에 넣음.
<img width="1041" alt="스크린샷 2021-07-16 오전 11 14 12" src="https://user-images.githubusercontent.com/58493928/125991464-9949b9a8-aeb8-4f17-af40-06465f0c83c6.png">
  - crop/warp 할 필요 없음.
- `Spatial Pyramid Matching`이란?
  - bag-of-words 와 비슷하게, bag-of-visual-words 개념 및 spatial 개념 도입: `위치(spatial) 정보를 바탕`으로, 이미지를 여러개로 쪼갠 뒤에 빈도수(히스토그램) 확인해서 새로운 매핑 정보로 변환. 
  - <img width="872" alt="스크린샷 2021-07-16 오전 11 26 25" src="https://user-images.githubusercontent.com/58493928/125992635-be3a1abd-a375-4421-b684-c604f879c9df.png">
  - <img width="1072" alt="스크린샷 2021-07-16 오전 11 40 16" src="https://user-images.githubusercontent.com/58493928/125994372-07d1a0b7-d978-4c98-b564-db292aa89782.png">
  - <img width="1072" alt="스크린샷 2021-07-16 오전 11 40 35" src="https://user-images.githubusercontent.com/58493928/125994408-5e237f0c-c1e9-44d9-b834-646ee47c3348.png">
