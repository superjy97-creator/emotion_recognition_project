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
        if not hasattr(sympy, 'core'):
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
    Jetson Orin Nano 기반 실시간 다중 얼굴 감정 인식 GUI 애플리케이션 Version 1.2
    (데드존 필터 기반 정지 화면 박스/감정 완전 고정 및 자원 강제 해제 적용)
    """
    def __init__(self, root):
        self.root = root
        self.root.title("얼굴 표정 기반 실시간 감정 인식(한국장애인고용공단 경기남부직업능력개발원) Version 1.2")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        target_w = int(screen_w * (2 / 3))
        target_h = int(screen_h)

        self.root.geometry(f"{target_w}x{target_h}+0+0")
        self.root.configure(bg="#1e1e1e")

        self.emotions = ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]

        self.class_weights = np.array([0.8, 1.2, 1.1, 1.2, 1.1, 1.1, 0.7], dtype=np.float32)
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
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
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
        self.video_frame = tk.Frame(self.root, bg="#000000")
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.video_label = tk.Label(self.video_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.control_frame = tk.Frame(self.root, bg="#2d2d2d", height=150)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)

        style = ttk.Style()
        style.theme_use('default')
        style.configure("TRadiobutton", background="#2d2d2d", foreground="#ffffff", font=("Malgun Gothic", 11, "bold"))

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

            # 데드존(Deadzone) 및 저역통과 필터(EMA) 결합: 미세 떨림 보정 및 이동 시 극도의 부드러움 보장
            if best_id in self.tracked_boxes_smooth:
                sx, sy, sw, sh = self.tracked_boxes_smooth[best_id]
                center_shift = np.hypot(cx - (sx + sw / 2.0), cy - (sy + sh / 2.0))
                size_diff = abs(w_i - sw) + abs(h_i - sh)

                # 중심점 이동 8px 미만 & 크기 변화 10px 미만 시 좌표 완벽 고정 (Deadzone 흡수)
                if center_shift < 8.0 and size_diff < 10:
                    sm_x, sm_y, sm_w, sm_h = sx, sy, sw, sh
                else:
                    # 박스 이동 시 급격한 튀기 현상 방지를 위해 계수(alpha)를 0.15로 최적화
                    alpha = 0.15
                    sm_x = int(alpha * x_i + (1 - alpha) * sx)
                    sm_y = int(alpha * y_i + (1 - alpha) * sy)
                    sm_w = int(alpha * w_i + (1 - alpha) * sw)
                    sm_h = int(alpha * h_i + (1 - alpha) * sh)
            else:
                sm_x, sm_y, sm_w, sm_h = x_i, y_i, w_i, h_i

            self.tracked_boxes_smooth[best_id] = (sm_x, sm_y, sm_w, sm_h)
            updated_tracked[best_id] = (sm_x, sm_y, sm_w, sm_h)
            self.face_lost_counter[best_id] = 0

        # 탐지 유실 시 버퍼 유지 프레임을 6 -> 12로 확장하여 짧은 깜빡임에도 ID와 상태가 초기화되지 않도록 개선
        for fid in list(self.tracked_faces.keys()):
            if fid not in assigned_ids:
                self.face_lost_counter[fid] = self.face_lost_counter.get(fid, 0) + 1
                if self.face_lost_counter[fid] <= 12:
                    updated_tracked[fid] = self.tracked_faces[fid]

        active_ids = set(updated_tracked.keys())
        self.prob_history = {fid: prob for fid, prob in self.prob_history.items() if fid in active_ids}
        self.prob_buffers = {fid: buf for fid, buf in self.prob_buffers.items() if fid in active_ids}
        self.emotion_states = {fid: st for fid, st in self.emotion_states.items() if fid in active_ids}
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
        """카메라 프레임과 추론을 분리하여 영상 재생 속도를 향상시키는 비동기 스레드"""
        while self.running:
            try:
                cv_img = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            self._process_detection_and_inference(cv_img)
            self.frame_queue.task_done()

    def _process_detection_and_inference(self, cv_img):
        """해상도 Downscale (0.5x) 탐지 및 데드존 기반 박스/신경망 입력 좌표 통합 동결"""
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
                
                # 핵심: 데드존/스무딩이 완벽히 고정된 좌표(sm_x, sm_y, sm_w, sm_h)로 모델 ROI 이미지 추출
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

                for idx, (sm_x, sm_y, sm_w, sm_h, face_id, color_rgb) in enumerate(valid_face_meta):
                    raw_probs = raw_probs_batch[idx]

                    calibrated_probs = raw_probs * self.class_weights
                    calibrated_probs = calibrated_probs / np.sum(calibrated_probs)

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

                    # 4) 순간 예측 감정 및 신뢰도 도출 (확률 마진 검증 추가)
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
                        
                        # 1, 2위 예측 간 격차가 8% 미만인 모호한 표정 구간에서는 기존 확정 감정 유지를 통해 감정 뒤바뀜 방지
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

                    if idx == 0:
                        self.current_emotion = final_emotion
                        self.current_confidence = raw_conf

                    new_detected_info.append((sm_x, sm_y, sm_w, sm_h, final_emotion, raw_conf, color_rgb, face_id))

            with self.detection_lock:
                self.detected_faces_info = new_detected_info
        else:
            with self.detection_lock:
                self.prob_history.clear()
                self.prob_buffers.clear()
                self.emotion_states.clear()
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
        draw = ImageDraw.Draw(pil_img)
        w, h = pil_img.size

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

            for (x, y, bw, bh, emotion, conf, color_rgb, face_id) in faces_to_draw:
                draw.rectangle([x, y, x + bw, y + bh], outline=color_rgb, width=3)

                tx = x + bw + 10
                ty = max(5, y + 2)
                
                #box_text = f"[ID:{face_id}] {emotion} ({conf:.1f}%)" if conf > 0 else f"[ID:{face_id}] {emotion}"
                box_text = f"{face_id}_{emotion} ({conf:.1f}%)" if conf > 0 else f"[ID:{face_id}] {emotion}"
                
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
        """실시간 FPS 비디오 출력 루프"""
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

            cv_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv_rgb)

            self.draw_overlays(pil_img)

            if self.is_saving_active:
                with self.detection_lock:
                    faces_snapshot = list(self.detected_faces_info)
                for face_info in faces_snapshot:
                    x, y, bw, bh, emotion, conf, color_rgb, face_id = face_info
                    if emotion not in ["미인식", "얼굴 미감지", "로딩중..."]:
                        save_key = (face_id, emotion)
                        if save_key not in self.saved_face_ids:
                            self._save_emotion_frame(pil_img, emotion, face_id)
                            self.saved_face_ids.add(save_key)

            lbl_w = self.video_label.winfo_width()
            lbl_h = self.video_label.winfo_height()
            
            if lbl_w > 100 and lbl_h > 100:
                render_w = max(200, int(lbl_w * 0.78125))
                render_h = max(150, int(lbl_h * 0.78125))
                pil_img_display = pil_img.resize((render_w, render_h), Image.BILINEAR)
            else:
                pil_img_display = pil_img

            imgtk = ImageTk.PhotoImage(image=pil_img_display)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        else:
            lbl_w = max(400, self.video_label.winfo_width())
            lbl_h = max(300, self.video_label.winfo_height())
            error_img = Image.new("RGB", (lbl_w, lbl_h), color=(30, 30, 30))
            draw = ImageDraw.Draw(error_img)
            draw.text((lbl_w // 2, lbl_h // 2), "카메라 영상 읽기 실패\n(웹캠 연결 및 장치 번호 확인 필요)", fill=(255, 87, 51), font=self.font_normal, anchor="mm")
            imgtk = ImageTk.PhotoImage(image=error_img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if self.running:
            self.after_id = self.root.after(10, self.update_video_frame)

    def on_close(self):
        """앱 종료 시 카메라 장치와 백그라운드 자원을 커널 레벨에서 즉시 해제"""
        print("[종료] 리소스 및 카메라 장치를 해제합니다...")
        self.running = False

        # 1. Tkinter 타이머 이벤트 즉시 취소
        if hasattr(self, 'after_id') and self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
                self.after_id = None
            except Exception:
                pass

        # 2. 비동기 프레임 큐 비우기
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except Exception:
            pass

        # 3. 카메라 디바이스 명시적 해제
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
                    print("[성공] 카메라 장치(/dev/video) 해제 완료")
            except Exception as e:
                print(f"[경고] 카메라 해제 중 오류: {e}")

        # 4. OpenCV 윈도우 파괴
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # 5. Tkinter GUI 파괴
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

        # 6. Jetson CUDA 백그라운드 드라이버 및 V4L2 점유 프로세스 완전 강제 종료
        print("[종료] 애플리케이션 완전 종료")
        os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = JetsonEmotionApp(root)
    root.mainloop()
 