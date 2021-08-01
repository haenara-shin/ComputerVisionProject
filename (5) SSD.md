# SSD
- Single-Shot Detector (One-stage detector)
- Region Proposal을 별도로 가지지 않음(별도로 가지는것들은 Two-stage detector, 앞서 공부 했던 것들. 느려서 실시간 영상/이미지 처리에 한계가 있음.)
- 수행 성능 및 속도를 모두 개선함. (Yolo가 버전업을 하고, RetinaNet(정밀도 up) 등장.)
- VGG16과 같은 backbone을 통과한 뒤에 생성된 feature map들 에서 추출한 anchor box들이 오브젝트에 대한 classification/detection을 같이 함.
  - ![스크린샷 2021-07-30 오전 11 01 02](https://user-images.githubusercontent.com/58493928/127693739-ebd2dd2f-a5d6-43a5-b97e-556b885402c6.png)
    - Classifier: Conv (4x(classes+4)) 의미 - anchor box 4개 * (클래스 갯수 + 백그라운드 1개 + 좌표값_anchor box와 gt box 사이의 offset 값(x, y, w, h), 즉 4) - w가 아니라 Center 인듯.
- 주요 구성 요소
  - Multi scale feature map(layer)
    - 슬라이딩 윈도우 remind! (슬라이딩 윈도우 크기를 크게 하면 IoU가 작아져서 detection 능력 저하)
    - 이미지 scale 조정에 따른(이미지 피라미드) 여러 크기의 object detection
    - 그렇다면, `서로 다른 크기의 feature map`을 이용한 object detection을 한다면? (CNN 통과후): cnn 통과하면서 점점 추상적인 특징을 가진 feature map을 얻게 됨. 
    - `feature map 사이즈가 작을수록`(추상화 잘 되어 있을 수록) 더 큰 오브젝트를 찾을 수 있음(`object detection 잘 됨`).
  - Default box(`anchor box` 와 같은 것)
    - 개별 anchor box가 다음 정보를 가질 수 있도록 학습
      - 개별 anchor box와 겹치는 feature map 영역의 object 클래스 분류(분류)
      - GT box 위치를 예측할 수 있도록 수정된 좌표(탐지)
    - 각 feature map의 크기에 대해서 개별 anchor box 별로 detection 하려는 Object 유형에 대한 softmax 값 및 수정 좌표 값을 학습하게 됨.
- SSD Training
  - Matching 전략: Bounding box와 겹치는 IoU가 0.5 이상인 default(anchor) box들의 classification과 bounding box regression을 최적화 학습 수행
  - Loss function
    - ![스크린샷 2021-07-30 오전 11 35 17](https://user-images.githubusercontent.com/58493928/127697152-a1b76d33-d8e7-46ff-8b34-438527263265.png)
  - 작은 오브젝트들이 잘 탐지가 안되는 경우 발생 --> Data augmentation 에 많은 노력 및 피처 피라미드/RetinaNet 등장

## SSD inference
- [Example code](https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API)
```python
import numpy as np
import tensorflow as tf
import cv2 as cv

# Read the graph.
with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    img = cv.imread('example.jpg')
    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

cv.imshow('TensorFlow MobileNet-SSD', img)
cv.waitKey()
```

## TensorFlowHub - pretrained model 사용 (SSD)
- [TensorFlowHub](https:/tfhub.dev) 
- CPU 에서 inference를 가능케 하려고 노력..했으나 성능은 별로
- 코드가 심플해짐!
- 주의 사항: tensorflow 에서는 bounding box 좌표를 [y_min, x_min, y_max, x_max] 순으로 반환/표현함.
```python
!pip install --upgrade tensorflow_hub

import tensorflow_hub as hub

model = hub.KerasLayer('https://tfhub.dev/google/nnlm-en-dim128/2')
embeddings = model(['something you want'])
print(embeddings.shape) # (4, 128)