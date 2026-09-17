# 얼굴 표정 기반 감정 인식 프로젝트

PyTorch 기반 딥러닝 모델을 활용하여 얼굴 표정에서 감정을 분류하고,  
NVIDIA Jetson Orin Nano 환경에서 실시간 카메라 입력을 통해 감정을 인식하는 프로젝트입니다.

---

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
NVIDIA Jetson Orin Nano에서 실행할 수 있도록 구성하였습니다.

---

## 2. 사용 기술

- Python 3.10
- PyTorch
- torchvision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow
- Jupyter Notebook
- Git / GitHub
- Git LFS
- NVIDIA Jetson Orin Nano

---

## 3. 사용 모델

본 프로젝트에서는 다음 3개의 CNN 모델을 비교하였습니다.

### MobileNetV3-Small

경량 구조를 기반으로 실시간 추론 및 엣지 디바이스 환경에 적합한 모델입니다.

### EfficientNet-B0

비교적 높은 분류 성능과 효율성을 함께 고려한 모델입니다.

### ResNet18

Residual Learning 구조를 활용하며 안정적인 이미지 분류 성능을 목표로 사용하였습니다.

각 모델은 학습 과정에서 다음 두 기준의 체크포인트를 저장하였습니다.

- Best Accuracy
- Best Loss

최종 학습에서는 RGB 컬러 JPG 데이터를 사용하여 MobileNetV3-Small, EfficientNet-B0,
ResNet18의 학습을 완료하였으며, 모델별 학습 결과와 3개 모델 비교 결과를 저장하였습니다.

---

## 4. 데이터셋

본 프로젝트에서는 AI-Hub의 「한국인 감정인식을 위한 복합 영상」 데이터를 활용하였습니다.

학습에 사용한 주요 데이터 구성은 다음과 같습니다.

- Training 이미지: 223,578장
- Validation 이미지: 52,126장
- 감정 클래스: 7개
- 입력 크기: 224×224

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
│  ├─ 프로젝트 일지
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
│  └─ 학습 완료 모델 및 Best 체크포인트 (.pt)
│
├─ 06_outputs/
│  ├─ Accuracy / Loss 그래프
│  ├─ Confusion Matrix
│  ├─ Precision / Recall / F1 그래프
│  ├─ Class Metrics CSV
│  ├─ Training History CSV
│  └─ 3개 모델 비교 결과 CSV / 그래프
│
├─ 07_app/
│  └─ jet_emotion_v012.py
│     └─ Jetson Orin Nano 실시간 다중 얼굴 감정 인식 실행 코드
│
├─ 08_demo_video/
│  ├─ 카메라 시연 영상1.mp4
│  ├─ 카메라 시연 영상2.mp4
│  └─ 프로젝트 발표 영상.mp4
│
├─ 09_reference/
│  └─ 프로젝트 참고 자료
│
├─ .gitattributes
├─ .gitignore
├─ requirements.txt
└─ README.md
```

---

## 6. 학습 및 평가 결과

최종 RGB 컬러 학습은 MobileNetV3-Small, EfficientNet-B0, ResNet18을 대상으로 진행하였습니다.

모델별로 다음 결과를 저장하였습니다.

- Training / Validation Accuracy
- Training / Validation Loss
- Confusion Matrix
- Precision
- Recall
- F1-score
- Class Metrics
- Training History
- Best Accuracy 체크포인트
- Best Loss 체크포인트

또한 세 모델의 성능을 비교하기 위한 종합 CSV와 비교 그래프를 `06_outputs`에 저장하였습니다.

---

## 7. 실시간 감정 인식 시스템

`07_app/jet_emotion_v012.py`는 NVIDIA Jetson Orin Nano에서 실행되는
실시간 다중 얼굴 감정 인식 프로그램입니다.

주요 기능은 다음과 같습니다.

- OpenCV 기반 실시간 카메라 입력
- Haar Cascade 기반 얼굴 탐지
- 다중 얼굴 ID 추적
- 얼굴 ROI 정사각형 크롭 및 224×224 변환
- Grayscale / RGB 입력 모드 지원
- EfficientNet-B0 / ResNet18 / MobileNetV3-Small 모델 선택
- PyTorch 기반 GPU 또는 CPU 추론
- 감정 확률 스무딩
- 감정 상태 유지
- 얼굴 오탐지 방지
- 바운딩 박스 / ID / 감정 결과 표시
- FPS 및 추론 장치 정보 표시

---

## 8. 데모 영상

`08_demo_video` 폴더에는 실시간 카메라 감정 인식 시연 영상과 프로젝트 발표 영상을 포함합니다.

- `카메라 시연 영상1.mp4`
- `카메라 시연 영상2.mp4`
- `프로젝트 발표 영상.mp4`

프로젝트 발표 영상은 파일 크기가 GitHub 일반 Git 파일 제한을 초과하므로
Git LFS(Large File Storage)를 사용하여 관리합니다.

---

## 9. 저장소 관리 원칙

- AI-Hub 원본 및 전처리 데이터는 GitHub에 업로드하지 않습니다.
- 학습 완료 모델과 평가 결과는 재현성과 결과 확인을 위해 저장합니다.
- 최종 프로젝트 문서는 `01_project_docs`에서 관리합니다.
- 대용량 프로젝트 발표 영상은 Git LFS로 관리합니다.
- 실험용 중간 파일보다 최종 학습 및 평가 산출물을 중심으로 저장소를 정리합니다.

---

## 10. 주요 산출물

본 저장소에는 다음과 같은 주요 프로젝트 산출물이 포함되어 있습니다.

- 프로젝트 수행계획서
- WBS
- 요구사항 명세서
- 프로젝트 일지
- 발표 자료 및 발표 대본
- 데이터 확인 / EDA / 전처리 Notebook
- PyTorch 모델 학습 Notebook
- MobileNetV3-Small 학습 모델
- EfficientNet-B0 학습 모델
- ResNet18 학습 모델
- 모델별 평가 결과
- 3개 모델 비교 결과
- Jetson Orin Nano 실시간 감정 인식 프로그램
- 카메라 시연 영상
- 프로젝트 발표 영상

---

## 11. 참고 사항

본 프로젝트는 교육 과정 내 팀 프로젝트로 진행되었으며,
대용량 데이터 처리, 이미지 전처리, 딥러닝 모델 학습, 모델 비교,
OpenCV 기반 실시간 영상 처리 및 Jetson Orin Nano 적용 과정을 포함합니다.