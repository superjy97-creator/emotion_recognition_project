# 얼굴 표정 기반 감정 인식 프로젝트

PyTorch 기반 딥러닝 모델을 활용하여 얼굴 표정에서 감정을 분류하고,  
NVIDIA Jetson Orin Nano 환경에서 실시간 카메라 입력을 통해 감정을 인식하는 프로젝트입니다.

## 1. 프로젝트 개요

본 프로젝트의 목표는 한국인 얼굴 표정 데이터를 활용하여 감정 분류 모델을 학습하고,
실시간 카메라 영상에서 얼굴을 탐지한 뒤 감정을 예측하여 화면에 표시하는 시스템을 구현하는 것입니다.

대상 감정 클래스는 총 7개입니다.

- 기쁨
- 당황
- 분노
- 불안
- 상처
- 슬픔
- 중립

실시간 시연 환경에서는 OpenCV 기반 얼굴 탐지와 PyTorch 모델 추론을 결합하였으며,
Jetson Orin Nano에서 실행할 수 있도록 구성하였습니다.

---

## 2. 사용 기술

- Python
- PyTorch
- torchvision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow
- Jupyter Notebook
- NVIDIA Jetson Orin Nano

---

## 3. 사용 모델

본 프로젝트에서는 다음 3개의 CNN 모델을 비교합니다.

### MobileNetV3-Small

경량 구조를 기반으로 실시간 추론 및 엣지 디바이스 환경에 적합한 모델입니다.

### EfficientNet-B0

비교적 높은 분류 성능과 효율성을 동시에 고려한 모델입니다.

### ResNet18

Residual Learning 구조를 사용하며 안정적인 이미지 분류 성능을 목표로 사용하였습니다.

학습 결과는 Best Accuracy 모델과 Best Loss 모델을 각각 저장하여 비교합니다.

---

## 4. 데이터셋

본 프로젝트에서는 AI-Hub의 한국인 감정 인식을 위한 복합 영상 데이터를 활용하였습니다.

학습에 사용한 주요 데이터 구성은 다음과 같습니다.

- Training 이미지: 223,578장
- Validation 이미지: 52,126장
- 감정 클래스: 7개

Training 데이터는 TRAIN_01과 TRAIN_02를 중심으로 구성하였으며,
이미지와 JSON 라벨을 연결하여 학습 데이터셋을 구성하였습니다.

> 데이터셋 원본 및 전처리 이미지 파일은 용량 및 데이터 이용 조건을 고려하여
> GitHub 저장소에 포함하지 않습니다.

---

## 5. 프로젝트 폴더 구조

```text
emotion_recognition_project/
├─ 01_project_docs/
│  ├─ 요구사항 명세서
│  ├─ WBS
│  ├─ 수행계획서
│  ├─ 발표 대본
│  └─ 발표 PPT
│
├─ 02_data/                    # 로컬 데이터 폴더 (GitHub 미포함)
│  ├─ downloads/
│  ├─ labels/
│  ├─ processed/
│  ├─ raw/
│  └─ sample/
│
├─ 03_notebooks/
│  └─ 데이터 확인, EDA, 전처리, 모델 학습 Notebook
│
├─ 05_models/
│  └─ 학습 완료 모델 (.pt)
│
├─ 06_outputs/
│  ├─ Accuracy / Loss 그래프
│  ├─ Confusion Matrix
│  ├─ Precision / Recall / F1 그래프
│  ├─ Class Metrics CSV
│  └─ Training History CSV
│
├─ 07_app/
│  └─ jet_emotion_v012.py
│     └─ Jetson Orin Nano 실시간 감정 인식 실행 코드
│
├─ 08_demo_video/
│  └─ 시연 영상
│
├─ 09_reference/
│  └─ 프로젝트 참고 자료
│
├─ .gitignore
├─ requirements.txt
└─ README.md