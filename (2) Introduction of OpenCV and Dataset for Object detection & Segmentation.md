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