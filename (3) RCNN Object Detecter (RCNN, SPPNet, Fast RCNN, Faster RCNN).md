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