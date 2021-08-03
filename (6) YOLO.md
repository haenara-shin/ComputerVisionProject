# YOLO (You Only Look Once)
## YOLO 각 버전 별 특징
### v1: 빠른 detection 시간, 낮은 정확도
- 입력 이미지를 s*s grid로(7x7) 나누고, '각 grid의 cell이 하나의 object에 대한 detection 수행'
- 각 grid cell이 2개의 bounding box 후보를 기반으로 object bounding box를 예측.
### v2: 수행 시간/성능 모두 개선 (SSD에 비해 작은 Object detection 성능 저하)
- SSD 다음 등장.
### v3: 수행 시간 조금 느리지만 성능 대폭 개선 (*)
- RetinaNet(FPN 채용) 다음 등장. Inference 속도가 굉장히 빠름.
### v4: 수행 시간 약간 개선, 성능 대폭 개선
- EfficientDet 다음 등장.