# 1. Project
![1](https://user-images.githubusercontent.com/50629716/68260828-875cab80-0081-11ea-8c9e-282c7916f433.png)
![2](https://user-images.githubusercontent.com/50629716/68260918-ca1e8380-0081-11ea-84f1-8d01e4b6a70d.png)


# 2. Introduction
행복과 화난 표정의 이미지(48x48)를 총 9150개를 학습시키고 총 1928개의 검증데이터를 CNN으로 학습 시켜 2가지의 감정을 분류 할 수 있게 만들었다.
데이터셋의 해상도가 낮아 모델의 층을 낮게 쌓았다.

# 3. Environment and Installation
## 1. 개발환경
* Anaconda3 설치
  * Python3.6x 버젼을 가상환경으로 추가.
* Pycharm 설치
  * Pycharm에서 Settings-> Project Interpreter에서 경로를 Anaconda3의 가상환경으로 추가한 Python3을 경로를 설정

## 2. 필수 라이브러리
* Keras
* numpy
* tensorflow
* cv2
* dlib
