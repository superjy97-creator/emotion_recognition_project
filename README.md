# 얼굴 표정 기반 감정 인식 프로젝트

PyTorch 기반 딥러닝 모델을 활용하여 얼굴 표정에서 감정을 분류하고,  
NVIDIA Jetson Orin Nano 환경에서 실시간 카메라 입력을 통해 감정을 인식하는 프로젝트입니다.

본 프로젝트는 팀 공동작업을 통해 데이터 확인, EDA, 전처리, 딥러닝 모델 학습, 모델 비교, 실시간 카메라 추론, Jetson Orin Nano 적용 및 전시용 프로그램 개선까지 진행하였습니다.

---

## 1. 프로젝트 개요

본 프로젝트의 목표는 한국인 얼굴 표정 데이터를 활용하여 감정 분류 모델을 학습하고, 실시간 카메라 영상에서 얼굴을 탐지한 뒤 감정을 예측하여 화면에 표시하는 시스템을 구현하는 것입니다.

대상 감정 클래스는 총 7개입니다.

- 기쁨
- 당황
- 분노
- 불안
- 상처
- 슬픔
- 중립

실시간 시연 환경에서는 OpenCV 기반 얼굴 탐지와 PyTorch 모델 추론을 결합하였으며, NVIDIA Jetson Orin Nano에서 실행할 수 있도록 구성하였습니다.

프로젝트 발표 이후에는 경기남부직업능력개발원 전시를 위한 추가 공동작업을 진행하였으며, 실제 카메라 인식 안정성을 고려하여 전시 환경에서는 RGB 컬러 모델을 중심으로 적용하였습니다.

---

## 2. 사용 기술

- Python 3.10 / 3.12
- PyTorch
- torchvision
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Pillow
- Jupyter Notebook
- Tkinter
- Git / GitHub
- Git LFS
- NVIDIA CUDA
- NVIDIA Jetson Orin Nano

프로젝트 진행 과정에서 작업 환경에 따라 Python 및 CUDA 구성에 일부 차이가 있었으며, 각 환경의 패키지 정보는 requirements 파일을 통해 관리하였습니다.

---

## 3. 학습 및 비교 모델

프로젝트 공동작업에서는 다양한 CNN 모델을 활용하여 감정 분류 성능을 비교하였습니다.

주요 모델은 다음과 같습니다.

### MobileNetV3-Small

경량 구조를 기반으로 실시간 추론 및 엣지 디바이스 환경에 적합한 모델입니다.

### EfficientNet-B0

비교적 높은 분류 성능과 효율성을 함께 고려한 모델입니다.

### ResNet18

Residual Learning 구조를 활용하며 안정적인 이미지 분류 성능을 목표로 사용하였습니다.

### MobileNetV2

공동작업물의 모델 비교 과정에서 활용한 경량 CNN 모델입니다.

각 모델은 학습 과정에서 Accuracy, Loss, Precision, Recall, F1-score 등의 지표를 기반으로 평가하였습니다.

학습 과정에서는 다음과 같은 기준의 체크포인트 및 결과물을 저장하였습니다.

- Best Accuracy
- Best Loss
- Best Model Checkpoint
- Training History
- Confusion Matrix
- Precision / Recall / F1-score
- 모델별 비교 결과

프로젝트 전체에는 RGB 컬러 JPG 기반 학습 결과와 흑백 PNG 기반 학습·평가 결과가 함께 포함되어 있으며, 모델별 성능 비교 자료와 평가 결과를 저장하였습니다.

---

## 4. 데이터셋

본 프로젝트에서는 AI-Hub의 「한국인 감정인식을 위한 복합 영상」 데이터를 활용하였습니다.

프로젝트 공동작업 과정에서 데이터 처리 방식과 학습 구성에 따라 다음과 같은 데이터셋을 사용하였습니다.

### RGB 컬러 학습 데이터

- Training 이미지: 223,578장
- Validation 이미지: 52,126장
- 감정 클래스: 7개
- 입력 크기: 224×224

Training 데이터는 TRAIN_01과 TRAIN_02를 중심으로 구성하였으며, 이미지와 JSON 라벨을 연결하여 학습 데이터셋을 구성하였습니다.

### 흑백 PNG 학습 데이터

- Training 이미지: 223,570장
- Validation 이미지: 52,118장
- 감정 클래스: 7개
- 입력 크기: 224×224
- 입력 형식: 흑백 PNG

원본 JPG와 JSON 라벨을 연결한 뒤 이미지 방향 보정과 224×224 변환을 거쳐 학습용 데이터를 구성하였습니다.

데이터 처리 방식과 유효 파일 검증 결과에 따라 학습 데이터 수에는 일부 차이가 있으며, 각 학습 결과는 해당 데이터 구성에 맞춰 평가하였습니다.

> AI-Hub 원본 데이터와 전처리 이미지 파일은 용량, 개인정보 및 데이터 이용 조건을 고려하여 GitHub 저장소에 포함하지 않습니다.

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
├─ 02_data/                          # 로컬 데이터 폴더 (GitHub 미포함)
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
│  └─ 최종 선정 모델 및 Best 체크포인트 (.pt)
│
├─ 06_outputs/
│  ├─ Accuracy / Loss 그래프
│  ├─ Confusion Matrix
│  ├─ Precision / Recall / F1 그래프
│  ├─ Class Metrics CSV
│  ├─ Training History CSV
│  └─ 모델 비교 결과 CSV / 그래프
│
├─ 07_app/
│  ├─ jet_emotion_v012.py
│  ├─ jet_emotion_v0131_fullscreen_touch.py
│  └─ jet_emotion_v0131_fullscreen_touch_titlebar.py
│
├─ 08_demo_video/
│  ├─ 카메라 시연 영상1.mp4
│  ├─ 카메라 시연 영상2.mp4
│  └─ 프로젝트 발표 영상.mp4
│
├─ 09_reference/
│  └─ 프로젝트 참고 자료
│
├─ 10_team_work/
│  └─ emotion_project-main/
│     ├─ docs/
│     ├─ results/
│     ├─ scripts/
│     ├─ src/
│     ├─ requirements.txt
│     ├─ requirements-common.lock.txt
│     ├─ requirements-cu126.lock.txt
│     ├─ .gitattributes
│     ├─ .gitignore
│     └─ README.md
│
├─ .gitattributes
├─ .gitignore
├─ requirements.txt
└─ README.md
```

`10_team_work/emotion_project-main`에는 프로젝트 공동작업물로 작성한 데이터 전처리, 모델 학습, 평가, 시각화, 환경 구성 및 결과 비교 관련 코드와 산출물을 정리하였습니다.

해당 폴더의 `models/`는 로컬 전달본에는 존재하지만, 현재 확보된 `best.pt` 파일이 실제 모델 바이너리가 아닌 Git LFS 포인터 파일이므로 GitHub 저장소에서는 제외하고 있습니다.

---

## 6. 학습 및 평가 결과

프로젝트에서는 여러 CNN 모델을 대상으로 학습 및 평가를 진행하였습니다.

저장된 주요 평가 결과는 다음과 같습니다.

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- Confusion Matrix
- Precision
- Recall
- F1-score
- Class Metrics
- Training History
- Best Accuracy 체크포인트
- Best Loss 체크포인트
- Best Model Checkpoint
- 모델별 비교 CSV
- 모델별 비교 그래프

`06_outputs` 폴더에는 흑백 2차 최종 학습·평가 결과와 RGB 컬러 최종 학습 결과, 그리고 모델 간 비교 결과가 저장되어 있습니다.

또한 `10_team_work/emotion_project-main/results`에는 공동작업물로 생성한 전처리 결과, 모델별 학습 결과, 세 모델 비교 결과 및 시각화 자료가 포함되어 있습니다.

---

## 7. 공동작업물

`10_team_work/emotion_project-main`에는 프로젝트 공동작업물로 작성한 데이터 전처리, 모델 학습, 평가, 비교, 환경 구성 및 결과 시각화 관련 코드와 산출물이 포함되어 있습니다.

공동작업물에서는 다음 모델을 활용하였습니다.

- EfficientNet-B0
- MobileNetV2
- ResNet18

데이터 구성은 다음과 같습니다.

- Training: 223,570장
- Validation: 52,118장
- 입력 이미지: 224×224 흑백 PNG
- 감정 클래스: 7개

주요 구성은 다음과 같습니다.

### 데이터 전처리

`src/preprocess_data.py`

- 원본 이미지 및 JSON 라벨 연결
- EXIF 방향 보정
- 224×224 흑백 이미지 생성
- 라벨 정리
- 전처리 결과 요약 생성

### 모델 학습

`src/train_models.py`

- EfficientNet-B0 학습
- MobileNetV2 학습
- ResNet18 학습
- GPU 기반 모델 학습
- Early Stopping
- 체크포인트 저장
- Training History 저장
- 실행 정보 저장

### 모델 비교

`src/visualize_results.py`

- 모델별 학습 결과 시각화
- Accuracy 비교
- F1-score 비교
- 클래스별 성능 비교
- 통합 비교표 생성

### 환경 확인

`src/check_environment.py`

- Python 버전 확인
- PyTorch 버전 확인
- CUDA 사용 여부 확인
- GPU 환경 확인

### CUDA 환경 구성

`scripts/setup_cuda126.ps1`

- Python 가상환경 구성
- CUDA 12.6 기반 PyTorch 환경 구성

### 데이터 전달

`src/package_processed_dataset.py`

- 전처리 데이터 전달용 압축 파일 생성

`src/unpack_processed_dataset.py`

- 전달받은 데이터 압축 파일 검증 및 복원

### 결과 저장

`results/experiments/`

- 모델별 Training History
- 실행 정보
- 평가 결과

`results/comparisons/all_three_models/`

- 세 모델 통합 비교 그래프
- 클래스별 F1-score
- 모델 비교표
- 결과 요약 자료

현재 전달된 `best.pt` 파일은 Git LFS 포인터 파일이므로 실제 모델 가중치는 GitHub에 포함하지 않았습니다.

---

## 8. 실시간 감정 인식 시스템

`07_app`에는 NVIDIA Jetson Orin Nano에서 실행하는 실시간 다중 얼굴 감정 인식 프로그램이 포함되어 있습니다.

### jet_emotion_v012.py

실시간 감정 인식 프로그램으로 다음 기능을 포함합니다.

- OpenCV 기반 실시간 카메라 입력
- Haar Cascade 기반 얼굴 탐지
- 다중 얼굴 ID 추적
- 얼굴 ROI 정사각형 크롭 및 224×224 변환
- Grayscale / RGB 입력 모드
- EfficientNet-B0 / ResNet18 / MobileNetV3-Small 모델 선택
- PyTorch 기반 GPU 또는 CPU 추론
- 감정 확률 스무딩
- 감정 상태 유지
- 얼굴 오탐지 방지
- 바운딩 박스 / ID / 감정 결과 표시
- FPS 및 추론 장치 정보 표시
- 감정 인식 결과 이미지 저장

### jet_emotion_v0131_fullscreen_touch.py

전시 환경을 고려하여 기능을 개선한 버전입니다.

주요 기능은 다음과 같습니다.

- 전체 화면 카메라 출력
- 화면 터치 또는 클릭 기반 설정 패널
- 원본 영상 비율 유지 최대 출력
- 감정별 클래스 가중치 조절
- 얼굴 바운딩 박스 흔들림 완화
- 감정 상태 표시 안정화
- 신뢰도 표시 안정화
- 스무딩 반응도 조절
- 상태 유지 프레임 조절
- Jetson 터치스크린 환경 대응

### jet_emotion_v0131_fullscreen_touch_titlebar.py

v1.3의 주요 기능을 유지하면서 운영 및 종료 편의성을 위해 타이틀바를 사용할 수 있도록 구성한 버전입니다.

---

## 9. 전시 적용

경기남부직업능력개발원 전시를 위해 팀 공동으로 실시간 감정 인식 프로그램의 추가 개선 작업을 진행하였습니다.

초기에는 Grayscale 모델과 RGB 컬러 모델을 모두 전시에 적용하는 방안을 검토하였습니다.

그러나 실제 Jetson 카메라 환경에서 Grayscale 모델의 감정 인식이 정상적으로 동작하지 않는 문제가 확인되었습니다.

이에 따라 전시에서는 안정적인 동작을 우선하여 RGB 컬러 모델을 중심으로 적용합니다.

전시 대응을 위해 다음 사항을 개선하였습니다.

- 실시간 카메라 영상 전체 화면 출력
- 터치 기반 설정 메뉴
- 얼굴 추적 안정화
- 바운딩 박스 떨림 감소
- 감정 예측 결과 유지
- 신뢰도 표시 안정화
- 감정별 가중치 조절
- 다중 얼굴 인식
- Jetson Orin Nano 환경 최적화
- 전시 환경에서의 조작 편의성 개선

---

## 10. 데모 영상

`08_demo_video` 폴더에는 실시간 카메라 감정 인식 시연 영상과 프로젝트 발표 영상을 포함합니다.

- `카메라 시연 영상1.mp4`
- `카메라 시연 영상2.mp4`
- `프로젝트 발표 영상.mp4`

프로젝트 발표 영상은 파일 크기가 GitHub 일반 Git 파일 제한을 초과하므로 Git LFS(Large File Storage)를 사용하여 관리합니다.

---

## 11. 저장소 관리 원칙

- AI-Hub 원본 및 전처리 데이터는 GitHub에 업로드하지 않습니다.
- 최종 선정 모델과 주요 평가 결과는 재현성과 결과 확인을 위해 관리합니다.
- 최종 프로젝트 문서는 `01_project_docs`에서 관리합니다.
- 대용량 프로젝트 발표 영상은 Git LFS로 관리합니다.
- 실험용 중간 파일보다 최종 학습 및 평가 산출물을 중심으로 저장소를 정리합니다.
- 공동작업 코드와 학습·평가 산출물은 `10_team_work`에서 관리합니다.
- `10_team_work/emotion_project-main/models/`의 `best.pt`는 현재 실제 모델 원본이 아닌 Git LFS 포인터만 확보된 상태이므로 GitHub 저장소에서는 제외합니다.
- 실제 모델 원본을 확보할 경우 Git LFS를 통해 추가 관리할 예정입니다.

---

## 12. 주요 산출물

본 저장소에는 다음과 같은 주요 프로젝트 산출물이 포함되어 있습니다.

- 프로젝트 수행계획서
- WBS
- 요구사항 명세서
- 프로젝트 일지
- 발표 자료 및 발표 대본
- 데이터 확인 Notebook
- EDA Notebook
- 전처리 Notebook
- PyTorch 모델 학습 Notebook
- MobileNetV3-Small 학습 모델
- EfficientNet-B0 학습 모델
- ResNet18 학습 모델
- 모델별 평가 결과
- 모델 비교 결과
- 공동 데이터 전처리 코드
- 공동 모델 학습 코드
- 공동 모델 평가 코드
- 공동 모델 비교 결과
- 환경 구성 스크립트
- Jetson Orin Nano 실시간 감정 인식 프로그램
- 전시용 전체 화면 / 터치 UI 프로그램
- 카메라 시연 영상
- 프로젝트 발표 영상

---

## 13. 참고 사항

본 프로젝트는 경기남부직업능력개발원 교육 과정 내 팀 프로젝트로 진행되었습니다.

프로젝트에서는 팀 공동작업을 통해 다음 과정을 수행하였습니다.

- 대용량 이미지 및 JSON 라벨 데이터 처리
- EDA 및 이미지 전처리
- PyTorch 기반 딥러닝 모델 학습
- 사전학습 가중치 활용
- GPU 기반 학습
- Accuracy / Loss / Precision / Recall / F1 평가
- 여러 CNN 모델의 성능 비교
- OpenCV 기반 실시간 영상 처리
- 다중 얼굴 탐지 및 추적
- 감정 예측 결과 안정화
- NVIDIA Jetson Orin Nano 적용
- Git / GitHub 기반 팀 산출물 관리
- Git LFS 기반 대용량 파일 관리
- 발표 이후 전시 환경을 위한 실시간 프로그램 개선

프로젝트 발표 완료 이후에도 경기남부직업능력개발원 전시 적용을 위해 실시간 카메라 프로그램의 UI와 인식 안정성을 팀 공동으로 지속 개선하고 있습니다.