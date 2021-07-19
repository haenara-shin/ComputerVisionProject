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

## Fast RCNN
<img width="1156" alt="스크린샷 2021-07-16 오후 12 00 13" src="https://user-images.githubusercontent.com/58493928/125996270-419c26fe-ceb1-43a7-826a-2ded2980dade.png">
- SPPNet과 유사한 `ROI Pooling layer` 존재.
  - feature map 상의 임의의 ROI를 고정 크기의 pooling 영역으로 매핑 (pool 크기를 일반적으로 7x7로 설정)
  - 매핑시 일반적으로 maxpooling 적용
  
![스크린샷 2021-07-16 오전 11 53 29](https://user-images.githubusercontent.com/58493928/125995585-7b3e9b16-116a-4d7b-b3b5-eda78cf42a52.png)
- 최종 layer는 softmax (SVM 아님)
- multi-task loss 함수로 classification과 regression을 함께 최적화
  - <img width="980" alt="스크린샷 2021-07-16 오후 12 04 13" src="https://user-images.githubusercontent.com/58493928/125996664-ed39685f-e7a0-4320-b6e0-3789e60ca650.png">

## Faster RCNN (딥러닝만으로 object detection 구현 시작)
<img width="1177" alt="스크린샷 2021-07-17 오후 1 13 31" src="https://user-images.githubusercontent.com/58493928/126048413-75d767af-c875-4f25-8c6e-34c1082738a0.png">
<img width="1010" alt="스크린샷 2021-07-17 오후 1 27 21" src="https://user-images.githubusercontent.com/58493928/126048701-09ad4f69-2d79-4fec-a225-cb51bacbac96.png">
<img width="1202" alt="스크린샷 2021-07-17 오후 1 28 33" src="https://user-images.githubusercontent.com/58493928/126048703-e187c784-266a-4801-a24e-2c41519481d4.png">

- CNN 통과한 feature map이 (1) RPN, (2) 그래도 RoI Pooling 진행
- RPN(Region Proposal Network, 전부 딥러닝으로 처리할 수 있게 됨. SPP를 대체함.) + Fast RCNN
  - RPN 구현 이슈: 데이터로 주어질 feature 는 pixel 값이고, target은 ground truth bounding box 인데 이를 어떻게 selective search 수준의 region proposal을 할 수 있을지?
    - `Anchor Box` 도입: Object가 있는지 없는지 후보 box. 총 9개의 `anchor box` (3개의 서로 다른 크기, 3개의 서로 다른 ratio 구성) --> object의 크기가 다양하기 때문.
  <img width="419" alt="스크린샷 2021-07-17 오후 1 17 40" src="https://user-images.githubusercontent.com/58493928/126048455-a5c82b66-4646-4c59-b280-7b060949a4f2.png">
  
    - 이미지와 feature map에서 anchor box 맵핑을 하고 backbone CNN(VGG)에 넣음.
    - ![스크린샷 2021-07-17 오후 1 30 27](https://user-images.githubusercontent.com/58493928/126048710-28359eb6-356f-4c7e-8c84-05f5c9f41bc9.png)
    - ![스크린샷 2021-07-17 오후 1 34 03](https://user-images.githubusercontent.com/58493928/126048799-39ce0a13-9329-4209-b8ea-2cb3821b1ceb.png)
  - `Positive anchor box`: Ground Truth BB 겹치는 IOU 값에 따라 Anchor box를 분류하는데, 0.7 이상이면 positive, 0.3 보다 낮으면 negative. (0.3~0.7은 애매하기 때문에 아예 학습에서 제외)
    - 예측 anchor box는 positive anchor box와의 좌표값 차이를 최소화 할 수 있는 bounding box regression 수행.
    - ![스크린샷 2021-07-17 오후 1 46 09](https://user-images.githubusercontent.com/58493928/126048988-13d7ca22-40e1-4fdc-b752-f439184066a1.png)
    - <img width="933" alt="스크린샷 2021-07-17 오후 1 48 45" src="https://user-images.githubusercontent.com/58493928/126049049-1fef8d09-11a6-4b5c-a68d-de1af3a0d41d.png">
- Summary <img width="1115" alt="스크린샷 2021-07-17 오후 1 53 48" src="https://user-images.githubusercontent.com/58493928/126049269-952370cf-1c57-428f-a6f7-a9114b2bd638.png">

## OpenCV로 Object Detection 구현
- OpenCV는 자체적으로 딥러닝 가중치 모델을 생성하지 않고 타 framework 에서 생성된 모델을 변환하여 로딩함.
- DNN 패키지는 파일로 생성된 타 framework 모델을 로딩할 수 있도록 `readNetFromXXX(가중치 모델 파일, 환경 설정 파일)` API를 제공함.
  - `가중치 모델 파일`: 타 framework 모델 파일
  - `환경 설정 파일`: 타 framework 모델 파일의 환경(config) 파일을 DNN 패키지에서 다시 변환한 환경 파일
```python
cvNet = Cv2.dnn.readNetFromTensorflow(가중치 모델 파일, 환경 파일)
``` 
  - [TensorFlow](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API)에서 가장 많은 유형의 detection/segmentation 모델 제공(가장 다양한 base network 지원). (그 외 Torch...)
    - 1. 가중치 모델 파일과 환경 설정 파일을 로드해서 inference network 모델 생성
      ```python
      cvNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb`, 'graph.pbtxt')
      img = cv2.imread('img.jpg')
      rows, cols, channels = img.shape
      ```
    - 2. 입력 이미지를 preprocessing 해서 network에 입력
      ```python
      cvNet.setInput(cv2.dnn.blobFromImage(img, size=(300,300),swapRB=True, crop=False))
      ```
      <img width="1131" alt="스크린샷 2021-07-18 오후 9 49 52" src="https://user-images.githubusercontent.com/58493928/126105099-667353c4-bad9-40d5-ac78-8c83f0f24323.png">

      <img width="1160" alt="스크린샷 2021-07-18 오후 9 50 28" src="https://user-images.githubusercontent.com/58493928/126105142-21533750-113c-4812-8e72-01e87d44377f.png">

    - 3. Inference network 에서 output 추출
      ```python
      networkOutput = cvNet.forward()
      ```
    - 4. 추출된 output 에서 detect 정보를 기반으로 원본 image 위에 object detection 시각화
      ```python
      for detection in networkOutput[0,0]:
        'object detected 된 결과, bounding box 좌표, 예측 레이블들을 원본 image 위에 시각화 로직'
  - Video 의 경우, `cv2.VideoCapture(input_file_path)`사용
    <img width="1160" alt="스크린샷 2021-07-18 오후 9 54 18" src="https://user-images.githubusercontent.com/58493928/126105179-fc22287a-6e6c-4e9b-930a-2344b58153eb.png">

  * 주의 사항!! <img width="847" alt="스크린샷 2021-07-18 오후 10 31 53" src="https://user-images.githubusercontent.com/58493928/126107933-aea76fa9-75de-4e04-826c-d17d9a58e4d5.png">

## Modern Object Detection Model Architecture
  <img width="923" alt="스크린샷 2021-07-18 오후 11 06 32" src="https://user-images.githubusercontent.com/58493928/126111457-490f24b8-a057-4517-9b38-0e160339afeb.png">

- `Backbone`: 원본 이미지 받아서 feature map 생성. Image classification model. (ex: ResNet, VGGNet..)
- `Neck`: Feature Pyramid Network (FPN). 앞선 backbone에서 만들어진 Feature map의 두께가 계속 증가(상세한 정보 --> 좀 더 추상화된 정보). 각 backbone layer에서 생성된 feature map을 모두 사용함 (각 feature map에서 담고 있는 정보가 모두 다를 수 있기 때문에. 예를 들어 동영상?. 즉, 작은 object 들을 보다 잘 detect하기 위해서 다양한 feature map 활용. 상위 feature map의 추상화된 정보와 하위 feature map의 정보를 효과적으로 결합) (이걸 건너 뛰고 그냥 bottom-up 각 단계에서 바로 detection 하는 경우가 `SSD`)
  <img width="878" alt="스크린샷 2021-07-18 오후 11 06 49" src="https://user-images.githubusercontent.com/58493928/126111535-77cfcbc3-d81e-477c-b591-cdd998c0ded0.png">
  
- <img width="930" alt="스크린샷 2021-07-18 오후 11 11 59" src="https://user-images.githubusercontent.com/58493928/126111617-ef2ef667-1a9d-4804-983a-0eff0bc78556.png">

- <img width="842" alt="스크린샷 2021-07-18 오후 11 12 51" src="https://user-images.githubusercontent.com/58493928/126111698-fba6abce-6dab-49a4-8da3-0290e9668276.png">

- <img width="934" alt="스크린샷 2021-07-18 오후 11 13 10" src="https://user-images.githubusercontent.com/58493928/126111750-7adcb87e-eea0-4907-8e4d-4e0eb617b9a5.png">