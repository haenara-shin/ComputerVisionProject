# Understanding of MMDetection & Faster RCNN
- `torchvision`: 지원 알고리즘 많지 않음..
- `Detectron2`: Config 기반. Facebook Research에서 주도.
- `MMDetection`: Config 기반. 중국 칭화 대학 중심의 OpenMMLab 주도. 지원되는 알고리즘 많음. 구현 성능이 뛰어남.

## MMDetection 개요
- [MMDetection: Open MMLab Detection Toolbox and Benchmark](https://arxiv.org/pdf/1906.07155.pdf)
- PyTorch 기반
- <img width="897" alt="스크린샷 2021-07-23 오전 10 57 40" src="https://user-images.githubusercontent.com/58493928/126822507-78a378c9-980a-44f8-8acf-e722acbf1894.png">
  
  - `MobileNet` 등이 없네..SSD와 같이 잘 쓰이는데 아쉬움. Yolo 버전도 낮음. `EfficientDet`은 구현 시도중? 확인 필요함.
- `Backbone`(FE) + `Neck`(FPN) + `Head`(DenseHead, AnchorHead/AnchorFreehead) + `RoIExtractor` + `RoIHead`(BBoxHead/MaskHead)
  - <img width="897" alt="스크린샷 2021-07-23 오전 10 57 40" src="https://user-images.githubusercontent.com/58493928/126822507-78a378c9-980a-44f8-8acf-e722acbf1894.png">
  
  - 1단계
    - `Backbone`: Feature extractor (이미지에서 feature map 뽑음)
    - `Neck`: Backbone과 Heads 를 연결하면서 Heads가 feature map의 특성을 보다 잘 해석하고 처리할 수 있도록 '정제 작업' 수행
    - `DenseHead`: Feature map 에서 object의 위치와 classification을 처리하는 부분
  - 2단계
    - `RoIExtractor`: Feature map에서 RoI정보를 뽑아내는 부분
    - `RoIHead(BBoxHead/MaskHead)`: RoI 정보를 기반으로 Object 위치와 classification을 수행하는 부분
- MMDetection 주요 구성 요소
  - <img width="811" alt="스크린샷 2021-07-23 오전 11 14 48" src="https://user-images.githubusercontent.com/58493928/126824480-a9492a86-e722-4e20-bae2-fe50c5f6ac7a.png">
- Config 기반의 구성
  - <img width="763" alt="스크린샷 2021-07-23 오전 11 18 26" src="https://user-images.githubusercontent.com/58493928/126824662-4de29a02-5f41-4b4d-a7a9-0410e1da86d3.png">
  - `rpn_head` 에서 클래스 갯수만 바꾸면 됨
  - `data`, `val` 주로 직접 개입. 
- MMDetection Training Pipeline
  - Hook(Callback이란 같은 것임)을 통해 학습에 필요한 여러 설정들을 customization 
  - 대부분 config 에서 설정함.
  - <img width="424" alt="스크린샷 2021-07-23 오전 11 18 44" src="https://user-images.githubusercontent.com/58493928/126824633-ed4411d0-5cd0-44d3-9079-4a01d0b5bd90.png">