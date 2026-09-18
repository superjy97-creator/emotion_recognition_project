import sys
import os
import time
import threading
import queue
import warnings
from datetime import datetime
from collections import deque

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont

# PyTorch 및 torchvision 세션 경고 억제
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import sympy
    if not hasattr(sympy, 'utilities'):
        import types
        sympy.utilities = types.ModuleType('utilities')
    if not hasattr(sympy.utilities, 'iterables'):
        import sympy.core.compatibility as compat
        sympy.utilities.iterables = compat
    if not hasattr(sympy, 'core') or not hasattr(sympy.core, 'sorting'):
        import types
        if not hasattr(sympy):
            sympy.core = types.ModuleType('core')
        sorting_mod = types.ModuleType('sorting')
        from sympy.utilities.iterables import ordered
        sorting_mod.ordered = ordered
        sympy.core.sorting = sorting_mod
except Exception:
    pass

if hasattr(np, '__version__') and np.__version__.startswith('2.'):
    print("[경고] NumPy 2.x 버전이 감지되었습니다. cv2 및 PyTorch ABI 충돌 발생 시 'pip3 install \"numpy<2\"'를 권장합니다.")

def find_model_file(possible_names):
    """
    스크립트 실행 위치 및 작업 디렉터리를 탐색하여 존재하는 가중치 파일의 절대 경로를 반환합니다.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cwd_dir = os.getcwd()
    search_dirs = [base_dir, cwd_dir]

    for name in possible_names:
        for s_dir in search_dirs:
            p = os.path.join(s_dir, name)
            if os.path.isfile(p):
                return p

    for name in possible_names:
        clean_keyword = name.replace(".pt", "").lower()
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                try:
                    for f in os.listdir(s_dir):
                        if f.endswith(".pt") and clean_keyword in f.lower():
                            return os.path.join(s_dir, f)
                except Exception:
                    pass
    return None

def get_resnet18(num_classes=7, in_channels=1):
    model = models.resnet18(weights=None)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_efficientnet_b0(num_classes=7, in_channels=1):
    model = models.efficientnet_b0(weights=None)
    if in_channels != 3:
        first_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels, first_conv.out_channels, 
            kernel_size=first_conv.kernel_size, 
            stride=first_conv.stride, 
            padding=first_conv.padding, 
            bias=False
        )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def get_mobilenet_v3(num_classes=7, in_channels=1):
    model = models.mobilenet_v3_small(weights=None)
    if in_channels != 3:
        first_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels, first_conv.out_channels, 
            kernel_size=first_conv.kernel_size, 
            stride=first_conv.stride, 
            padding=first_conv.padding, 
            bias=False
        )
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

class JetsonEmotionApp:
    """
    Jetson Orin Nano 기반 실시간 다중 얼굴 감정 인식 GUI 애플리케이션 Version 1.3
    (전체 화면 카메라 출력, 영상 터치형 설정 패널, 원본 비율 유지 최대 출력)
    """
    def __init__(self, root):
        self.root = root
        self.root.title("얼굴 표정 기반 실시간 감정 인식(한국장애인고용공단 경기남부직업능력개발원) Version 1.3")

        # Jetson이 현재 출력 중인 실제 화면 해상도를 사용합니다.
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()

        # 기본 화면은 완전한 전체 화면으로 시작합니다.
        # 따라서 타이틀바/창 테두리 없이 카메라 영상 영역이 Jetson 출력 화면 전체를 사용합니다.
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.maxsize(self.screen_w, self.screen_h)
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#000000")

        # 설정 패널은 기본적으로 숨김 상태입니다.
        # 카메라 영상을 터치(마우스 클릭 포함)할 때만 화면 아래쪽에 오버레이로 표시됩니다.
        self.controls_visible = False

        self.emotions = ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]
        
        # 클래스 불균형 보정 기본 가중치
        #self.default_weights = [0.8, 1.2, 1.1, 1.2, 1.1, 1.1, 0.7]
        self.default_weights = [1.2, 0.8, 0.8, 0.7, 0.7, 0.8, 1.1]
        self.class_min_thresh = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[정보] 추론 연산 디바이스: {self.device}")

        self.face_cascades = self._get_enhanced_face_detectors()

        self.transform_gray = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.models_loaded = False
        self.current_emotion = "로딩중..."
        self.current_confidence = 0.0

        self.frame_queue = queue.Queue(maxsize=1)
        self.detection_lock = threading.Lock()

        self.prob_history = {}
        self.prob_buffers = {}
        self.emotion_states = {}

        # 화면에 표시되는 신뢰도는 추론값과 별도로 안정화합니다.
        # 정지 영상에서도 센서 노이즈나 얼굴 검출 ROI의 미세 변화로 확률이 흔들리는 현상을 줄입니다.
        self.display_confidence_states = {}
        
        self.tracked_faces = {}
        self.tracked_boxes_smooth = {}
        self.face_lost_counter = {}
        self.next_face_id = 0
        self.detected_faces_info = []

        self.is_saving_active = False
        self.saved_face_ids = set()

        self.smooth_alpha_var = tk.DoubleVar(value=0.15)
        self.hysteresis_frames_var = tk.IntVar(value=5)
        self.window_size = 8

        # 카메라 요청 해상도. 실제 카메라가 다른 해상도로 출력하면 실제 프레임 크기를 사용합니다.
        self.cam_base_w = 800
        self.cam_base_h = 600

        # 화면 출력은 고정 배율이 아니라, 비디오 영역 크기에 맞춰 원본 종횡비를 유지하며 최대화합니다.
        self.render_w = self.cam_base_w
        self.render_h = self.cam_base_h
        self.render_scale_x = 1.0
        self.render_scale_y = 1.0

        self.model_configs = {
            "EfficientNet": {
                "file_gray_candidates": ["be_efficientnet.pt", "be_efficientnet_b0.pt", "efficientnet.pt"],
                "file_color_candidates": ["c_efficientnet.pt", "c_efficientnet_b0.pt", "EfficientNet-B0-c_efficientnet.pt"],
                "file_gray": "be_efficientnet.pt",
                "file_color": "c_efficientnet.pt",
                "builder": get_efficientnet_b0,
                "haar_scale": 1.10,
                "min_neighbors": 5,
                "temperature": 1.0,
                "clahe": True,
                "smooth_factor": 0.15,
                "margin_ratio": 0.14,
                "desc": "독립적 개별 감정 고정밀 인식 모드"
            },
            "ResNet": {
                "file_gray_candidates": ["be_resnet.pt", "be_resnet18.pt", "resnet.pt"],
                "file_color_candidates": ["c_resnet.pt", "c_resnet18.pt", "ResNet18-c_resnet.pt"],
                "file_gray": "be_resnet.pt",
                "file_color": "c_resnet.pt",
                "builder": get_resnet18,
                "haar_scale": 1.10,
                "min_neighbors": 5,
                "temperature": 1.0,
                "clahe": True,
                "smooth_factor": 0.15,
                "margin_ratio": 0.14,
                "desc": "조명 변화에 강인한 표정 인식 모드"
            },
            "MobileNet": {
                "file_gray_candidates": ["be_mobilenet.pt", "be_mobilenet_v3.pt", "mobilenet.pt"],
                "file_color_candidates": ["c_mobilenet.pt", "c_mobilenet_v3.pt", "MobileNetV3-Small-c_mobilenet.pt"],
                "file_gray": "be_mobilenet.pt",
                "file_color": "c_mobilenet.pt",
                "builder": get_mobilenet_v3,
                "haar_scale": 1.10,
                "min_neighbors": 5,
                "temperature": 1.0,
                "clahe": True,
                "smooth_factor": 0.15,
                "margin_ratio": 0.14,
                "desc": "다중 객체 초고속 실시간 인식 모드"
            }
        }

        self.loaded_models = {"gray": {}, "color": {}}
        self.model_status = {"gray": {}, "color": {}}

        self.selected_color_mode = tk.StringVar(value="컬러")
        self.selected_model_name = tk.StringVar(value="EfficientNet")

        self.switch_msg = ""
        self.switch_msg_until = 0.0

        self.selected_color_mode.trace_add("write", self._on_mode_or_model_changed)
        self.selected_model_name.trace_add("write", self._on_mode_or_model_changed)

        self._build_ui()

        self.font_normal = self._load_korean_font(size=22)
        self.font_large = self._load_korean_font(size=34)
        self.font_huge = self._load_korean_font(size=66)

        self.cap = self._init_camera()

        self.root.bind("<q>", lambda e: self.on_close())
        self.root.bind("<Q>", lambda e: self.on_close())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        threading.Thread(target=self._load_selected_model, daemon=True).start()

        self.running = True
        self.prev_time = time.time()
        self.fps = 30.0
        self.after_id = None

        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inference_thread.start()

        self.update_video_frame()

    def _get_enhanced_face_detectors(self):
        cascade_names = [
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_frontalface_default.xml"
        ]
        
        base_search_paths = [
            "/usr/share/opencv4/haarcascades/",
            "/usr/share/opencv/haarcascades/",
            "/usr/local/share/opencv4/haarcascades/",
            "./"
        ]
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            base_search_paths.insert(0, cv2.data.haarcascades)

        loaded_cascades = []
        for cname in cascade_names:
            for bpath in base_search_paths:
                full_p = os.path.join(bpath, cname)
                if os.path.exists(full_p):
                    c = cv2.CascadeClassifier(full_p)
                    if not c.empty():
                        loaded_cascades.append(c)
                        print(f"[성공] 탐지기 로드: {full_p}")
                        break

        if not loaded_cascades:
            loaded_cascades.append(cv2.CascadeClassifier())

        return loaded_cascades

    def _load_korean_font(self, size=22):
        font_paths = [
            "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def _init_camera(self):
        cap = None
        for dev_idx in [0, 1, 2]:
            temp_cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
            if not temp_cap.isOpened():
                temp_cap = cv2.VideoCapture(dev_idx)
            
            if temp_cap.isOpened():
                ret, test_frame = temp_cap.read()
                if ret and test_frame is not None:
                    cap = temp_cap
                    break
                else:
                    temp_cap.release()

        if cap is None:
            return cv2.VideoCapture(0)

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_base_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_base_h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print(f"[경고] 카메라 설정 변경: {e}")

        return cap

    def _get_random_vibrant_color(self, idx):
        palette = [
            (255, 87, 51), (51, 255, 87), (51, 161, 255), (255, 219, 51),
            (220, 51, 255), (51, 255, 220), (255, 140, 0), (255, 105, 180)
        ]
        return palette[idx % len(palette)]

    def _build_ui(self):
        # 비디오 프레임은 루트 창 전체를 항상 차지합니다.
        # 설정 패널이 나타나도 pack 영역을 줄이지 않고 영상 위에 겹쳐 표시되도록 구성합니다.
        self.video_frame = tk.Frame(self.root, bg="#000000")
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        # 비디오 라벨은 고정 크기를 사용하지 않습니다.
        # 전체 화면 안에서 카메라 원본 종횡비를 유지한 최대 크기로 렌더링합니다.
        self.video_label = tk.Label(
            self.video_frame,
            bg="#000000",
            bd=0,
            highlightthickness=0,
            cursor="hand2"
        )
        self.video_label.pack(expand=True, anchor=tk.CENTER)

        # 영상 자체와 영상 주변의 검은 여백 모두 터치 대상으로 사용합니다.
        # 터치스크린의 일반적인 탭 이벤트는 Tkinter에서 <Button-1>로 전달됩니다.
        self.video_label.bind("<Button-1>", self._toggle_controls)
        self.video_frame.bind("<Button-1>", self._toggle_controls)

        # 설정 패널은 영상 영역의 크기를 변경하지 않는 오버레이입니다.
        # _toggle_controls()가 호출될 때 화면 하단에 place()로 표시됩니다.
        self.control_frame = tk.Frame(
            self.root,
            bg="#2d2d2d",
            height=190,
            bd=2,
            relief=tk.RAISED
        )

        style = ttk.Style()
        style.theme_use('default')
        style.configure("TRadiobutton", background="#2d2d2d", foreground="#ffffff", font=("Malgun Gothic", 11, "bold"))

        # 1행: 색상 선택 및 모델 선택
        self.row_color_frame = tk.Frame(self.control_frame, bg="#2d2d2d")
        self.row_color_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(6, 2))

        lbl_color = tk.Label(self.row_color_frame, text="색상 선택:", bg="#2d2d2d", fg="#ffcc00", font=("Malgun Gothic", 11, "bold"))
        lbl_color.pack(side=tk.LEFT, padx=(5, 5))

        color_options = [
            ("컬러 (Color RGB)", "컬러"),
            ("그레이 (Grayscale)", "그레이")
        ]

        for text, mode in color_options:
            rb_c = ttk.Radiobutton(
                self.row_color_frame, 
                text=text, 
                value=mode, 
                variable=self.selected_color_mode,
                style="TRadiobutton"
            )
            rb_c.pack(side=tk.LEFT, padx=8)

        lbl_model = tk.Label(self.row_color_frame, text=" |  모델 선택:", bg="#2d2d2d", fg="#00ffcc", font=("Malgun Gothic", 11, "bold"))
        lbl_model.pack(side=tk.LEFT, padx=(15, 5))

        model_options = [
            ("EfficientNet", "EfficientNet"),
            ("ResNet", "ResNet"),
            ("MobileNet", "MobileNet")
        ]

        for text, mode in model_options:
            rb_m = ttk.Radiobutton(
                self.row_color_frame, 
                text=text, 
                value=mode, 
                variable=self.selected_model_name,
                style="TRadiobutton"
            )
            rb_m.pack(side=tk.LEFT, padx=8)

        self.lbl_desc = tk.Label(self.row_color_frame, text="", bg="#2d2d2d", fg="#aaaaaa", font=("Malgun Gothic", 10, "italic"))
        self.lbl_desc.pack(side=tk.LEFT, padx=10)
        self.selected_model_name.trace_add("write", lambda *args: self._update_model_desc())
        self._update_model_desc()

        btn_exit = tk.Button(
            self.row_color_frame,
            text="종료",
            command=self.on_close,
            bg="#ff3333",
            fg="#ffffff",
            activebackground="#cc0000",
            activeforeground="#ffffff",
            font=("Malgun Gothic", 10, "bold"),
            width=7,
            relief=tk.RAISED,
            bd=2
        )
        btn_exit.pack(side=tk.RIGHT, padx=5)

        # 2행: 스무딩 및 상태 유지 프레임 조절
        self.row_smooth_frame = tk.Frame(self.control_frame, bg="#2d2d2d")
        self.row_smooth_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)

        lbl_smooth_title = tk.Label(self.row_smooth_frame, text="변동성 완화:", bg="#2d2d2d", fg="#00e5ff", font=("Malgun Gothic", 11, "bold"))
        lbl_smooth_title.pack(side=tk.LEFT, padx=(5, 5))

        lbl_alpha = tk.Label(self.row_smooth_frame, text="스무딩 반응도(EMA):", bg="#2d2d2d", fg="#ffffff", font=("Malgun Gothic", 9))
        lbl_alpha.pack(side=tk.LEFT, padx=(5, 2))

        slider_alpha = tk.Scale(
            self.row_smooth_frame,
            from_=0.05,
            to=0.50,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.smooth_alpha_var,
            bg="#2d2d2d",
            fg="#00e5ff",
            highlightthickness=0,
            length=130,
            troughcolor="#444444"
        )
        slider_alpha.pack(side=tk.LEFT, padx=(0, 15))

        lbl_hold = tk.Label(self.row_smooth_frame, text="상태 유지 프레임 수:", bg="#2d2d2d", fg="#ffffff", font=("Malgun Gothic", 9))
        lbl_hold.pack(side=tk.LEFT, padx=(5, 2))

        slider_hold = tk.Scale(
            self.row_smooth_frame,
            from_=1,
            to=10,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.hysteresis_frames_var,
            bg="#2d2d2d",
            fg="#00e5ff",
            highlightthickness=0,
            length=120,
            troughcolor="#444444"
        )
        slider_hold.pack(side=tk.LEFT, padx=(0, 10))

        lbl_smooth_hint = tk.Label(
            self.row_smooth_frame, 
            text="(낮을수록 표정이 부드럽고 안정적으로 유지됨)", 
            bg="#2d2d2d", 
            fg="#888888", 
            font=("Malgun Gothic", 9, "italic")
        )
        lbl_smooth_hint.pack(side=tk.LEFT, padx=5)

        # 3행: 감정별 클래스 가중치 (Class Weights) 입력 패널
        self.row_weight_frame = tk.Frame(self.control_frame, bg="#2d2d2d")
        self.row_weight_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)

        lbl_weight_title = tk.Label(self.row_weight_frame, text="감정 가중치:", bg="#2d2d2d", fg="#ff66cc", font=("Malgun Gothic", 11, "bold"))
        lbl_weight_title.pack(side=tk.LEFT, padx=(5, 5))

        self.weight_vars = []
        for idx, emotion in enumerate(self.emotions):
            lbl_e = tk.Label(self.row_weight_frame, text=f"{emotion}:", bg="#2d2d2d", fg="#ffffff", font=("Malgun Gothic", 9))
            lbl_e.pack(side=tk.LEFT, padx=(4, 1))
            
            var = tk.StringVar(value=str(self.default_weights[idx]))
            self.weight_vars.append(var)
            
            ent = tk.Entry(
                self.row_weight_frame, 
                textvariable=var, 
                width=5, 
                bg="#3d3d3d", 
                fg="#00e5ff", 
                insertbackground="#ffffff", 
                font=("Malgun Gothic", 9, "bold"),
                justify="center"
            )
            ent.pack(side=tk.LEFT, padx=(0, 6))

        btn_reset_weights = tk.Button(
            self.row_weight_frame,
            text="초기화",
            command=self.reset_class_weights,
            bg="#555555",
            fg="#ffffff",
            activebackground="#777777",
            activeforeground="#ffffff",
            font=("Malgun Gothic", 8, "bold"),
            width=6,
            relief=tk.RAISED,
            bd=1
        )
        btn_reset_weights.pack(side=tk.LEFT, padx=5)

        # 4행: 데이터 저장 제어
        self.row_save_frame = tk.Frame(self.control_frame, bg="#2d2d2d")
        self.row_save_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(2, 6))

        lbl_save = tk.Label(self.row_save_frame, text="데이터 저장:", bg="#2d2d2d", fg="#ff9900", font=("Malgun Gothic", 11, "bold"))
        lbl_save.pack(side=tk.LEFT, padx=(5, 10))

        self.btn_start_save = tk.Button(
            self.row_save_frame,
            text="저장 시작",
            command=self.start_saving,
            bg="#28a745",
            fg="#ffffff",
            activebackground="#218838",
            activeforeground="#ffffff",
            font=("Malgun Gothic", 9, "bold"),
            width=9,
            relief=tk.RAISED,
            bd=2
        )
        self.btn_start_save.pack(side=tk.LEFT, padx=4)

        self.btn_stop_save = tk.Button(
            self.row_save_frame,
            text="저장 종료",
            command=self.stop_saving,
            bg="#dc3545",
            fg="#ffffff",
            activebackground="#c82333",
            activeforeground="#ffffff",
            font=("Malgun Gothic", 9, "bold"),
            width=9,
            relief=tk.RAISED,
            bd=2
        )
        self.btn_stop_save.pack(side=tk.LEFT, padx=4)

        self.lbl_save_status = tk.Label(self.row_save_frame, text="[저장 중지됨]", bg="#2d2d2d", fg="#ff4444", font=("Malgun Gothic", 10, "bold"))
        self.lbl_save_status.pack(side=tk.LEFT, padx=12)

    def _toggle_controls(self, event=None):
        """
        카메라 영상을 터치/클릭할 때 설정 패널을 표시하거나 숨깁니다.

        - 기본 상태: 카메라 영상만 전체 화면 표시
        - 첫 번째 터치: 화면 하단에 설정/버튼 패널 표시
        - 다시 영상 터치: 설정 패널 숨김
        """
        if self.controls_visible:
            self.control_frame.place_forget()
            self.controls_visible = False
        else:
            # 영상 위에 겹쳐 표시하므로 카메라 렌더링 영역은 계속 전체 화면을 유지합니다.
            self.control_frame.place(
                relx=0.0,
                rely=1.0,
                relwidth=1.0,
                anchor="sw"
            )
            self.control_frame.lift()
            self.controls_visible = True

    def reset_class_weights(self):
        for idx, def_val in enumerate(self.default_weights):
            if idx < len(self.weight_vars):
                self.weight_vars[idx].set(str(def_val))

    def get_current_class_weights(self):
        weights = []
        for idx, var in enumerate(self.weight_vars):
            try:
                val = float(var.get())
                weights.append(val)
            except (ValueError, tk.TclError):
                weights.append(self.default_weights[idx])
        return np.array(weights, dtype=np.float32)

    def start_saving(self):
        self.is_saving_active = True
        self.lbl_save_status.config(text="[저장 동작 중 (ID별 1회)]", fg="#00ff66")

    def stop_saving(self):
        self.is_saving_active = False
        self.lbl_save_status.config(text="[저장 중지됨]", fg="#ff4444")

    def _update_model_desc(self):
        active_name = self.selected_model_name.get()
        cfg = self.model_configs.get(active_name, {})
        desc_text = f"[{cfg.get('desc', '')}]" if cfg else ""
        if hasattr(self, 'lbl_desc'):
            self.lbl_desc.config(text=desc_text)

    def _on_mode_or_model_changed(self, *args):
        c_mode = self.selected_color_mode.get()
        m_name = self.selected_model_name.get()
        
        with self.detection_lock:
            self.prob_history.clear()
            self.prob_buffers.clear()
            self.emotion_states.clear()
            self.display_confidence_states.clear()
            self.tracked_faces.clear()
            self.tracked_boxes_smooth.clear()
            self.detected_faces_info.clear()
            self.saved_face_ids.clear()
        
        self.switch_msg = f"[{c_mode} - {m_name}] 모델 로딩 및 전환 중..."
        self.switch_msg_until = time.time() + 2.0

        threading.Thread(target=self._load_selected_model, daemon=True).start()

    def _load_selected_model(self):
        c_mode_str = self.selected_color_mode.get()
        active_model_name = self.selected_model_name.get()
        mode_key = "color" if c_mode_str == "컬러" else "gray"
        in_ch = 3 if mode_key == "color" else 1

        cfg = self.model_configs.get(active_model_name, self.model_configs["EfficientNet"])
        candidates_key = "file_color_candidates" if mode_key == "color" else "file_gray_candidates"
        candidate_names = cfg.get(candidates_key, [cfg["file_color" if mode_key == "color" else "file_gray"]])

        if mode_key not in self.model_status:
            self.model_status[mode_key] = {}
        self.model_status[mode_key][active_model_name] = "LOADING"

        target_path = find_model_file(candidate_names)
        model_fn = cfg["builder"]
        model = model_fn(num_classes=len(self.emotions), in_channels=in_ch).to(self.device)
        weights_loaded_success = False

        if target_path:
            try:
                checkpoint = torch.load(target_path, map_location=self.device)
                state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
                if isinstance(state_dict, dict):
                    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
                    model.load_state_dict(new_state_dict, strict=False)
                    weights_loaded_success = True
            except Exception:
                weights_loaded_success = False

        model.eval()

        if mode_key not in self.loaded_models:
            self.loaded_models[mode_key] = {}

        self.loaded_models[mode_key][active_model_name] = model
        self.model_status[mode_key][active_model_name] = weights_loaded_success
        self.models_loaded = True
        self.current_emotion = "얼굴 미감지"

    def _track_faces(self, current_faces):
        updated_tracked = {}
        assigned_ids = set()

        for (x, y, w, h) in current_faces:
            x_i, y_i, w_i, h_i = int(x), int(y), int(w), int(h)
            cx, cy = x_i + w_i / 2.0, y_i + h_i / 2.0
            best_id = None
            max_iou_score = -1.0

            for fid, (prev_x, prev_y, prev_w, prev_h) in self.tracked_faces.items():
                if fid in assigned_ids:
                    continue
                
                inter_x1 = max(x_i, prev_x)
                inter_y1 = max(y_i, prev_y)
                inter_x2 = min(x_i + w_i, prev_x + prev_w)
                inter_y2 = min(y_i + h_i, prev_y + prev_h)
                
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                
                area1 = w_i * h_i
                area2 = prev_w * prev_h
                union_area = area1 + area2 - inter_area
                iou = inter_area / float(union_area + 1e-6)

                pcx, pcy = prev_x + prev_w / 2.0, prev_y + prev_h / 2.0
                center_dist = np.hypot(cx - pcx, cy - pcy)

                score = iou * 2.0 - (center_dist / 200.0)
                if (iou > 0.20 or center_dist < 130.0) and score > max_iou_score:
                    max_iou_score = score
                    best_id = fid

            if best_id is None:
                best_id = self.next_face_id
                self.next_face_id += 1

            assigned_ids.add(best_id)

            if best_id in self.tracked_boxes_smooth:
                sx, sy, sw, sh = self.tracked_boxes_smooth[best_id]

                # Haar 검출기는 정지된 얼굴에서도 몇 픽셀씩 좌표/크기가 흔들릴 수 있습니다.
                # 얼굴 크기에 비례한 데드존 안의 변화는 실제 움직임이 아닌 검출 노이즈로 보고
                # 이전 박스를 그대로 사용합니다. 이렇게 하면 정지 영상에서 사각형이 변형되지 않습니다.
                prev_cx = sx + sw / 2.0
                prev_cy = sy + sh / 2.0
                center_shift = np.hypot(cx - prev_cx, cy - prev_cy)
                width_diff = abs(w_i - sw)
                height_diff = abs(h_i - sh)

                center_deadzone = max(6.0, min(sw, sh) * 0.035)
                size_deadzone = max(8.0, min(sw, sh) * 0.05)

                if (center_shift <= center_deadzone and
                        width_diff <= size_deadzone and
                        height_diff <= size_deadzone):
                    # 미세한 좌표/크기 변화는 완전히 고정
                    sm_x, sm_y, sm_w, sm_h = sx, sy, sw, sh
                else:
                    # 실제 이동이 감지된 경우에만 천천히 따라갑니다.
                    # 작은 이동은 강하게 평활화하고, 큰 이동은 조금 더 빠르게 반응합니다.
                    move_ratio = center_shift / max(float(min(sw, sh)), 1.0)
                    alpha = 0.08 if move_ratio < 0.12 else 0.18

                    sm_x = int(round(alpha * x_i + (1.0 - alpha) * sx))
                    sm_y = int(round(alpha * y_i + (1.0 - alpha) * sy))
                    sm_w = max(1, int(round(alpha * w_i + (1.0 - alpha) * sw)))
                    sm_h = max(1, int(round(alpha * h_i + (1.0 - alpha) * sh)))
            else:
                sm_x, sm_y, sm_w, sm_h = x_i, y_i, w_i, h_i

            self.tracked_boxes_smooth[best_id] = (sm_x, sm_y, sm_w, sm_h)
            updated_tracked[best_id] = (sm_x, sm_y, sm_w, sm_h)
            self.face_lost_counter[best_id] = 0

        for fid in list(self.tracked_faces.keys()):
            if fid not in assigned_ids:
                self.face_lost_counter[fid] = self.face_lost_counter.get(fid, 0) + 1
                if self.face_lost_counter[fid] <= 12:
                    updated_tracked[fid] = self.tracked_faces[fid]

        active_ids = set(updated_tracked.keys())
        self.prob_history = {fid: prob for fid, prob in self.prob_history.items() if fid in active_ids}
        self.prob_buffers = {fid: buf for fid, buf in self.prob_buffers.items() if fid in active_ids}
        self.emotion_states = {fid: st for fid, st in self.emotion_states.items() if fid in active_ids}
        self.display_confidence_states = {
            fid: st for fid, st in self.display_confidence_states.items() if fid in active_ids
        }
        self.tracked_boxes_smooth = {fid: box for fid, box in self.tracked_boxes_smooth.items() if fid in active_ids}
        self.face_lost_counter = {fid: cnt for fid, cnt in self.face_lost_counter.items() if fid in active_ids}
        self.tracked_faces = updated_tracked
        return updated_tracked

    def _crop_and_pad_square(self, img, x1, y1, x2, y2):
        h, w = img.shape[:2]
        crop_w = x2 - x1
        crop_h = y2 - y1

        max_dim = max(crop_w, crop_h)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        new_x1 = max(0, cx - max_dim // 2)
        new_y1 = max(0, cy - max_dim // 2)
        new_x2 = min(w, cx + max_dim // 2)
        new_y2 = min(h, cy + max_dim // 2)

        return img[new_y1:new_y2, new_x1:new_x2].copy()

    def _save_emotion_frame(self, pil_img, emotion, face_id):
        save_dir = "Data"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        now_str = datetime.now().strftime("%Y%m%d%H%M%S")
        seq_num = 1

        while True:
            filename = f"{emotion}_ID{face_id}_{now_str}_{seq_num}.png"
            filepath = os.path.join(save_dir, filename)
            if not os.path.exists(filepath):
                break
            seq_num += 1

        try:
            pil_img.save(filepath, format="PNG")
            print(f"[저장 완료] 인식 ID [{face_id}] 감정({emotion}) 저장: {filepath}")
        except Exception as e:
            print(f"[저장 오류] 이미지 저장 중 오류: {e}")

    def _inference_worker(self):
        while self.running:
            try:
                cv_img = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            self._process_detection_and_inference(cv_img)
            self.frame_queue.task_done()

    def _process_detection_and_inference(self, cv_img):
        img_h, img_w = cv_img.shape[:2]
        
        scale = 0.5
        small_w = int(img_w * scale)
        small_h = int(img_h * scale)
        small_gray = cv2.resize(cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY), (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        
        color_mode_str = self.selected_color_mode.get()
        mode_key = "color" if color_mode_str == "컬러" else "gray"
        
        active_model_name = self.selected_model_name.get()
        cfg = self.model_configs.get(active_model_name, self.model_configs["EfficientNet"])
        model = self.loaded_models.get(mode_key, {}).get(active_model_name, None)
        status = self.model_status.get(mode_key, {}).get(active_model_name, False)

        is_loading = (status == "LOADING")

        haar_scale = cfg.get("haar_scale", 1.10)
        min_neighbors = cfg.get("min_neighbors", 5)
        temp_scale = cfg.get("temperature", 1.0)
        use_clahe = cfg.get("clahe", True)
        margin_ratio = cfg.get("margin_ratio", 0.14)

        smooth_factor = float(self.smooth_alpha_var.get())
        required_hold_frames = int(self.hysteresis_frames_var.get())

        raw_faces_small = []
        for detector in self.face_cascades:
            faces = detector.detectMultiScale(
                small_gray, 
                scaleFactor=haar_scale, 
                minNeighbors=min_neighbors, 
                minSize=(30, 30)
            )
            if len(faces) > 0:
                raw_faces_small = list(faces)
                break

        if len(raw_faces_small) == 0:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_small_gray = clahe.apply(small_gray)
            for detector in self.face_cascades:
                faces = detector.detectMultiScale(
                    enhanced_small_gray,
                    scaleFactor=1.08,
                    minNeighbors=max(4, min_neighbors - 1),
                    minSize=(30, 30)
                )
                if len(faces) > 0:
                    raw_faces_small = list(faces)
                    break

        valid_faces_small = []
        for (x, y, w, h) in raw_faces_small:
            aspect_ratio = float(w) / float(h)
            if 0.65 <= aspect_ratio <= 1.35:
                roi = small_gray[y:y+h, x:x+w]
                if roi.size > 0 and np.std(roi) > 14.0:
                    valid_faces_small.append((x, y, w, h))

        raw_faces_small = valid_faces_small
        inv_scale = 1.0 / scale
        raw_faces = [[int(x * inv_scale), int(y * inv_scale), int(w * inv_scale), int(h * inv_scale)] for (x, y, w, h) in raw_faces_small]

        if len(raw_faces) > 1:
            boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in raw_faces])
            scores = np.array([w * h for (x, y, w, h) in raw_faces], dtype=np.float32)
            indices = cv2.dnn.NMSBoxes(
                bboxes=[[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in boxes],
                scores=scores.tolist(),
                score_threshold=0.0,
                nms_threshold=0.3
            )
            if len(indices) > 0:
                idx_flat = np.array(indices).flatten()
                raw_faces = [raw_faces[i] for i in idx_flat]

        new_detected_info = []

        if is_loading:
            self.current_emotion = "모델 로딩 중..."
            self.current_confidence = 0.0

        if len(raw_faces) > 0 and model is not None and self.models_loaded and not is_loading:
            raw_faces = sorted(raw_faces, key=lambda f: f[0])
            
            with self.detection_lock:
                tracked_dict = self._track_faces(raw_faces)
            
            face_id_map = {(int(x), int(y), int(w), int(h)): fid for fid, (x, y, w, h) in tracked_dict.items()}

            face_tensors = []
            valid_face_meta = []
            gray_full = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if mode_key == "gray" else None

            for idx, (x, y, w, h) in enumerate(raw_faces):
                x_i, y_i, w_i, h_i = int(x), int(y), int(w), int(h)
                matched_fid = face_id_map.get((x_i, y_i, w_i, h_i), self.next_face_id + idx)
                
                matched_box = tracked_dict.get(matched_fid, (x_i, y_i, w_i, h_i))
                sm_x, sm_y, sm_w, sm_h = matched_box

                color_rgb = self._get_random_vibrant_color(matched_fid)

                margin_w = int(sm_w * margin_ratio)
                margin_h = int(sm_h * margin_ratio)
                x1 = max(0, sm_x - margin_w)
                y1 = max(0, sm_y - margin_h)
                x2 = min(img_w, sm_x + sm_w + margin_w)
                y2 = min(img_h, sm_y + sm_h + margin_h)

                if mode_key == "gray":
                    face_roi = self._crop_and_pad_square(gray_full, x1, y1, x2, y2)
                    if face_roi.size == 0:
                        face_roi = gray_full[sm_y:sm_y+sm_h, sm_x:sm_x+sm_w].copy()

                    if use_clahe:
                        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        face_roi = clahe_op.apply(face_roi)

                    input_tensor = self.transform_gray(face_roi)
                else:
                    face_roi_bgr = self._crop_and_pad_square(cv_img, x1, y1, x2, y2)
                    if face_roi_bgr.size == 0:
                        face_roi_bgr = cv_img[sm_y:sm_y+sm_h, sm_x:sm_x+sm_w].copy()

                    face_roi_rgb = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2RGB)
                    input_tensor = self.transform_color(face_roi_rgb)

                face_tensors.append(input_tensor)
                valid_face_meta.append((sm_x, sm_y, sm_w, sm_h, matched_fid, color_rgb))

            if len(face_tensors) > 0:
                batch_tensor = torch.stack(face_tensors, dim=0).to(self.device)
                
                with torch.no_grad():
                    logits = model(batch_tensor)
                    scaled_logits = logits / temp_scale
                    raw_probs_batch = torch.softmax(scaled_logits, dim=1).cpu().numpy()

                current_weights = self.get_current_class_weights()

                for idx, (sm_x, sm_y, sm_w, sm_h, face_id, color_rgb) in enumerate(valid_face_meta):
                    raw_probs = raw_probs_batch[idx]

                    calibrated_probs = raw_probs * current_weights
                    calibrated_probs = calibrated_probs / (np.sum(calibrated_probs) + 1e-7)

                    with self.detection_lock:
                        if face_id not in self.prob_buffers:
                            self.prob_buffers[face_id] = deque(maxlen=self.window_size)
                        self.prob_buffers[face_id].append(calibrated_probs)

                        buf_arr = np.array(self.prob_buffers[face_id])
                        weights = np.linspace(0.5, 1.0, len(buf_arr))
                        window_avg_probs = np.average(buf_arr, axis=0, weights=weights)

                        if face_id in self.prob_history:
                            prev_probs = self.prob_history[face_id]
                            smoothed_probs = (1.0 - smooth_factor) * prev_probs + smooth_factor * window_avg_probs
                        else:
                            smoothed_probs = window_avg_probs
                        
                        self.prob_history[face_id] = np.copy(smoothed_probs)

                    sorted_indices = np.argsort(smoothed_probs)[::-1]
                    raw_max_idx = sorted_indices[0]
                    second_max_idx = sorted_indices[1]

                    raw_conf = float(smoothed_probs[raw_max_idx]) * 100.0
                    second_conf = float(smoothed_probs[second_max_idx]) * 100.0
                    conf_margin = raw_conf - second_conf
                    candidate_emotion = self.emotions[raw_max_idx]

                    min_req_conf = self.class_min_thresh[raw_max_idx]
                    if raw_conf < min_req_conf:
                        candidate_emotion = "미인식"

                    with self.detection_lock:
                        if face_id not in self.emotion_states:
                            self.emotion_states[face_id] = {
                                "locked_emotion": candidate_emotion,
                                "candidate": candidate_emotion,
                                "count": 0
                            }

                        st = self.emotion_states[face_id]
                        
                        if candidate_emotion != st["locked_emotion"] and conf_margin < 8.0:
                            candidate_emotion = st["locked_emotion"]

                        if candidate_emotion == st["locked_emotion"]:
                            st["candidate"] = candidate_emotion
                            st["count"] = 0
                        else:
                            if candidate_emotion == st["candidate"]:
                                st["count"] += 1
                                if st["count"] >= required_hold_frames:
                                    st["locked_emotion"] = candidate_emotion
                                    st["count"] = 0
                            else:
                                st["candidate"] = candidate_emotion
                                st["count"] = 1

                        final_emotion = st["locked_emotion"]

                        # 화면 표시용 신뢰도는 현재 잠금된 감정의 확률을 기준으로 별도 안정화합니다.
                        # 기존 코드는 매 프레임 최고 확률(raw_conf)을 그대로 표시해서,
                        # 정지 상태에서도 숫자가 계속 바뀌어 보일 수 있었습니다.
                        if final_emotion in self.emotions:
                            final_idx = self.emotions.index(final_emotion)
                            target_conf = float(smoothed_probs[final_idx]) * 100.0
                        else:
                            target_conf = raw_conf

                        conf_state = self.display_confidence_states.get(face_id)
                        if conf_state is None or conf_state.get("emotion") != final_emotion:
                            # 감정 자체가 바뀐 경우에는 새로운 감정의 신뢰도로 즉시 초기화
                            display_conf = target_conf
                        else:
                            prev_display_conf = float(conf_state.get("confidence", target_conf))
                            conf_diff = target_conf - prev_display_conf

                            # 1%p 이내 변화는 센서/ROI 노이즈로 간주하여 표시값 고정
                            if abs(conf_diff) <= 1.0:
                                display_conf = prev_display_conf
                            else:
                                # 변화가 필요한 경우에도 천천히 이동시켜 숫자 떨림을 억제
                                conf_alpha = 0.08
                                display_conf = prev_display_conf + conf_alpha * conf_diff

                        display_conf = float(np.clip(display_conf, 0.0, 100.0))
                        self.display_confidence_states[face_id] = {
                            "emotion": final_emotion,
                            "confidence": display_conf
                        }

                    if idx == 0:
                        self.current_emotion = final_emotion
                        self.current_confidence = display_conf

                    new_detected_info.append((
                        sm_x, sm_y, sm_w, sm_h, final_emotion, display_conf, color_rgb, face_id
                    ))

            with self.detection_lock:
                self.detected_faces_info = new_detected_info
        else:
            with self.detection_lock:
                self.prob_history.clear()
                self.prob_buffers.clear()
                self.emotion_states.clear()
                self.display_confidence_states.clear()
                self.tracked_faces.clear()
                self.tracked_boxes_smooth.clear()
                self.face_lost_counter.clear()
                self.detected_faces_info.clear()
                if not self.models_loaded:
                    self.current_emotion = "로딩중..."
                    self.current_confidence = 0.0
                else:
                    self.current_emotion = "얼굴 미감지"
                    self.current_confidence = 0.0

    def draw_overlays(self, pil_img):
        """
        현재 비디오 표시 크기에 맞춰 오버레이를 렌더링합니다.
        카메라 원본 종횡비를 유지한 상태에서 영상이 확대/축소되므로,
        얼굴 박스 좌표도 실제 렌더링 배율에 맞춰 동일하게 보정합니다.
        """
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size # 1000, 750

        if self.models_loaded:
            dot_color = (0, 255, 0)
            for r in range(3):
                for c in range(3):
                    x0 = 10 + c * 8
                    y0 = 10 + r * 8
                    draw.ellipse([x0, y0, x0 + 5, y0 + 5], fill=dot_color)

        if time.time() < self.switch_msg_until:
            box_w, box_h = 540, 50
            x_min, y_min = (w - box_w) // 2, 20
            x_max, y_max = (w + box_w) // 2, 20 + box_h
            draw.rectangle([x_min, y_min, x_max, y_max], fill=(0, 0, 0, 220), outline=(0, 255, 204), width=2)
            draw.text((w // 2, y_min + box_h // 2), self.switch_msg, fill=(0, 255, 204), font=self.font_normal, anchor="mm")
        elif len(self.detected_faces_info) == 0:
            top_em_color = (255, 255, 0)
            emotion_str = f"상태: {self.current_emotion}"
            draw.text((w // 2, 45), emotion_str, fill=top_em_color, font=self.font_large, anchor="mm")

        with self.detection_lock:
            faces_to_draw = list(self.detected_faces_info)

        if len(faces_to_draw) > 0:
            active_font = self.font_large
            sx = self.render_scale_x
            sy = self.render_scale_y

            for (x, y, bw, bh, emotion, conf, color_rgb, face_id) in faces_to_draw:
                # 얼굴 좌표를 실제 화면 렌더링 크기에 맞춰 변환합니다.
                disp_x = int(x * sx)
                disp_y = int(y * sy)
                disp_bw = int(bw * sx)
                disp_bh = int(bh * sy)

                draw.rectangle([disp_x, disp_y, disp_x + disp_bw, disp_y + disp_bh], outline=color_rgb, width=3)

                tx = disp_x + disp_bw + 10
                ty = max(5, disp_y + 2)
                
                # 글자 크기는 기존 원본 크기 유지
                box_text = f"{face_id}_{emotion} ({conf:.1f}%)" if conf > 0 else f"{face_id}_{emotion}"
                
                try:
                    bbox = draw.textbbox((tx, ty), box_text, font=active_font)
                    draw.rectangle(
                        [bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4], 
                        fill=(0, 0, 0, 200),
                        outline=color_rgb,
                        width=2
                    )
                except Exception:
                    pass

                draw.text((tx, ty), box_text, fill=color_rgb, font=active_font)

        lime_green = (51, 255, 87)
        active_color = self.selected_color_mode.get()
        active_model = self.selected_model_name.get()
        save_state_str = "저장: ON" if self.is_saving_active else "저장: OFF"
        alpha_val = self.smooth_alpha_var.get()
        hold_val = self.hysteresis_frames_var.get()
        
        model_info_text = f"모드: {active_color} | 모델: {active_model} | α:{alpha_val:.2f} | Hold:{hold_val}F | {save_state_str}"
        draw.text((20, h - 50), model_info_text, fill=lime_green, font=self.font_normal, anchor="lb")

        device_str = "GPU (CUDA)" if self.device.type == "cuda" else "CPU"
        device_info_text = f"추론 연산 디바이스: {device_str} ({int(round(self.fps))} FPS)"
        draw.text((20, h - 15), device_info_text, fill=lime_green, font=self.font_normal, anchor="lb")

    def update_video_frame(self):
        if not self.running:
            return

        curr_time = time.time()
        dt = curr_time - self.prev_time
        self.prev_time = curr_time
        if dt > 0:
            self.fps = 0.85 * self.fps + 0.15 * (1.0 / dt)

        ret, frame = self.cap.read()
        if ret and frame is not None:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

            # 1) 현재 비디오 영역에 맞춰 카메라 영상의 종횡비를 유지하면서 최대 크기를 계산합니다.
            cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img_raw = Image.fromarray(cv_rgb)
            src_w, src_h = pil_img_raw.size

            # pack/layout 계산이 완료된 실제 비디오 컨테이너 크기를 사용합니다.
            container_w = max(1, self.video_frame.winfo_width())
            container_h = max(1, self.video_frame.winfo_height())

            # 초기 배치 순간 1x1이 반환되는 경우에는 안전한 기본 크기를 사용합니다.
            if container_w <= 2 or container_h <= 2:
                container_w = min(src_w, self.screen_w)
                container_h = min(src_h, self.screen_h)

            # 종횡비 유지: 가로/세로 중 먼저 닿는 축을 기준으로 동일 배율 적용
            fit_scale = min(container_w / float(src_w), container_h / float(src_h))
            render_w = max(1, int(src_w * fit_scale))
            render_h = max(1, int(src_h * fit_scale))

            self.render_w = render_w
            self.render_h = render_h
            self.render_scale_x = render_w / float(src_w)
            self.render_scale_y = render_h / float(src_h)

            pil_img_scaled = pil_img_raw.resize((render_w, render_h), Image.BILINEAR)

            # 2) 실제 렌더링 배율에 맞춰 얼굴 박스 및 텍스트 오버레이를 합성합니다.
            self.draw_overlays(pil_img_scaled)

            if self.is_saving_active:
                with self.detection_lock:
                    faces_snapshot = list(self.detected_faces_info)
                for face_info in faces_snapshot:
                    x, y, bw, bh, emotion, conf, color_rgb, face_id = face_info
                    if emotion not in ["미인식", "얼굴 미감지", "로딩중..."]:
                        save_key = (face_id, emotion)
                        if save_key not in self.saved_face_ids:
                            self._save_emotion_frame(pil_img_scaled, emotion, face_id)
                            self.saved_face_ids.add(save_key)

            # 3) 계산된 최대 크기로 렌더링합니다. 창 크기가 바뀌면 다음 프레임부터 자동 재계산됩니다.
            imgtk = ImageTk.PhotoImage(image=pil_img_scaled)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        else:
            # 카메라 오류 화면도 현재 비디오 영역을 넘지 않도록 동적으로 생성합니다.
            error_w = max(320, min(self.video_frame.winfo_width(), self.screen_w))
            error_h = max(240, min(self.video_frame.winfo_height(), self.screen_h))
            error_img = Image.new("RGB", (error_w, error_h), color=(30, 30, 30))
            draw = ImageDraw.Draw(error_img)
            draw.text(
                (error_w // 2, error_h // 2), 
                "카메라 영상 읽기 실패\n(웹캠 연결 및 장치 번호 확인 필요)", 
                fill=(255, 87, 51), 
                font=self.font_normal, 
                anchor="mm"
            )
            imgtk = ImageTk.PhotoImage(image=error_img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if self.running:
            self.after_id = self.root.after(10, self.update_video_frame)

    def on_close(self):
        print("[종료] 리소스 및 카메라 장치를 해제합니다...")
        self.running = False

        if hasattr(self, 'after_id') and self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
                self.after_id = None
            except Exception:
                pass

        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except Exception:
            pass

        if hasattr(self, 'cap') and self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
                    print("[성공] 카메라 장치(/dev/video) 해제 완료")
            except Exception as e:
                print(f"[경고] 카메라 해제 중 오류: {e}")

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

        print("[종료] 애플리케이션 완전 종료")
        os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = JetsonEmotionApp(root)
    root.mainloop()
