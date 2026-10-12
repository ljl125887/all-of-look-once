"""车辆/人员实例查找与四路实时监控独立验证版 GUI。"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from core import InstanceSearchPipeline, SearchConfig, format_time, read_image
from live import LiveMonitorManager, StreamConfig
from profiles import TargetLibrary, TargetProfile
from reid_models import PERSON_MODEL_FUSION, PERSON_MODEL_OPTIONS


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "instance_search_results"
TARGET_LIBRARY_PATH = APP_DIR / "data" / "targets"


class CropDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc, image_source: Path | object, category: str = "目标"):
        super().__init__(parent)
        self.title(f"框选同一个具体{category} — 按住鼠标左键拖动")
        self.transient(parent)
        self.grab_set()
        self.result = None
        self.image_bgr = image_source.copy() if hasattr(image_source, "shape") else read_image(image_source)
        h, w = self.image_bgr.shape[:2]
        max_w = max(640, self.winfo_screenwidth() - 180)
        max_h = max(480, self.winfo_screenheight() - 220)
        self.scale = min(1.0, max_w / w, max_h / h)
        self.display_w = max(1, round(w * self.scale))
        self.display_h = max(1, round(h * self.scale))
        rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        preview = Image.fromarray(rgb).resize((self.display_w, self.display_h), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(preview)
        self.canvas = tk.Canvas(self, width=self.display_w, height=self.display_h, bg="#07101f", cursor="cross")
        self.canvas.pack(padx=12, pady=(12, 6))
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.start_xy: tuple[int, int] | None = None
        self.rect_id: int | None = None
        self.box: tuple[int, int, int, int] | None = None
        self.canvas.bind("<ButtonPress-1>", self._start)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._finish)

        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Label(bar, text=f"请尽量贴合{category}边缘，减少无关背景。可以重复拖动重新框选。").pack(side="left")
        ttk.Button(bar, text="取消", command=self.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(bar, text="确认目标", command=self._confirm).pack(side="right")
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _clamp(self, x: int, y: int) -> tuple[int, int]:
        return max(0, min(self.display_w, x)), max(0, min(self.display_h, y))

    def _start(self, event) -> None:
        self.start_xy = self._clamp(event.x, event.y)
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        x, y = self.start_xy
        self.rect_id = self.canvas.create_rectangle(x, y, x, y, outline="#34d399", width=4)

    def _drag(self, event) -> None:
        if self.start_xy is None or self.rect_id is None:
            return
        x, y = self._clamp(event.x, event.y)
        self.canvas.coords(self.rect_id, self.start_xy[0], self.start_xy[1], x, y)

    def _finish(self, event) -> None:
        if self.start_xy is None:
            return
        x, y = self._clamp(event.x, event.y)
        x1, x2 = sorted((self.start_xy[0], x))
        y1, y2 = sorted((self.start_xy[1], y))
        self.box = (x1, y1, x2, y2)

    def _confirm(self) -> None:
        if self.box is None or self.box[2] - self.box[0] < 15 or self.box[3] - self.box[1] < 15:
            messagebox.showwarning("尚未框选", "请先用鼠标框选目标车辆。", parent=self)
            return
        x1, y1, x2, y2 = (round(value / self.scale) for value in self.box)
        h, w = self.image_bgr.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        self.result = self.image_bgr[y1:y2, x1:x2].copy()
        self.destroy()


class InstanceSearchApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("车辆/人员实例查找验证版 — 离线视频 + 四路实时监控")
        self.geometry("1500x920")
        self.minsize(1180, 760)
        self.configure(bg="#07101f")
        self.references: list = []
        self.reference_photos: list[ImageTk.PhotoImage] = []
        self.pipeline: InstanceSearchPipeline | None = None
        self.target_library = TargetLibrary(TARGET_LIBRARY_PATH)
        self.current_profile_id: str | None = None
        self.live_manager: LiveMonitorManager | None = None
        self.live_running = False
        self.pending_offline: tuple[SearchConfig, list] | None = None
        self.live_photos: dict[int, ImageTk.PhotoImage] = {}
        self.global_tree_rows: dict[str, str] = {}
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.messages: queue.Queue = queue.Queue()
        self.last_output: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self._configure_style()
        self._build_ui()
        self._load_config()
        self.after(80, self._poll_messages)
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background="#101b2d", foreground="#dbeafe", fieldbackground="#17243a", font=("Microsoft YaHei UI", 10))
        style.configure("TFrame", background="#101b2d")
        style.configure("Card.TFrame", background="#101b2d", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#101b2d", foreground="#dbeafe")
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 16, "bold"), foreground="#60a5fa")
        style.configure("Hint.TLabel", foreground="#93a4bd")
        style.configure("TButton", padding=(12, 7))
        style.configure("Primary.TButton", background="#34d399", foreground="#06281f", font=("Microsoft YaHei UI", 10, "bold"))
        style.map("Primary.TButton", background=[("active", "#6ee7b7"), ("disabled", "#3c5b56")])
        style.configure("Danger.TButton", background="#fb7185", foreground="#341016")
        style.configure("Treeview", background="#111e32", fieldbackground="#111e32", foreground="#dbeafe", rowheight=29)
        style.configure("Treeview.Heading", background="#263650", foreground="#e5efff", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("Horizontal.TProgressbar", background="#34d399", troughcolor="#17243a")

    def _build_ui(self) -> None:
        header = ttk.Frame(self, padding=(18, 14))
        header.pack(fill="x")
        ttk.Label(header, text="车辆/人员实例查找 · 离线视频 + 四路实时监控", style="Title.TLabel").pack(side="left")
        self.device_label = ttk.Label(header, text="模型尚未加载", style="Hint.TLabel")
        self.device_label.pack(side="right")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        offline_tab = ttk.Frame(self.notebook)
        live_tab = ttk.Frame(self.notebook)
        self.notebook.add(offline_tab, text="离线视频实例查找")
        self.notebook.add(live_tab, text="四路实时视频监控")

        body = ttk.Panedwindow(offline_tab, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, style="Card.TFrame", padding=14)
        right = ttk.Frame(body, style="Card.TFrame", padding=10)
        body.add(left, weight=1)
        body.add(right, weight=3)

        ttk.Label(left, text="1. 登记同一个具体实例", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        category_row = ttk.Frame(left)
        category_row.pack(fill="x", pady=(10, 2))
        ttk.Label(category_row, text="目标类别", width=9).pack(side="left")
        self.category_var = tk.StringVar(value="车辆")
        ttk.Combobox(category_row, textvariable=self.category_var, values=("车辆", "人员"), state="readonly", width=12).pack(side="left", fill="x", expand=True)
        row = ttk.Frame(left)
        row.pack(fill="x", pady=(4, 6))
        ttk.Label(row, text="目标名称", width=9).pack(side="left")
        self.target_var = tk.StringVar(value="目标车辆")
        ttk.Entry(row, textvariable=self.target_var).pack(side="left", fill="x", expand=True)

        ref_buttons = ttk.Frame(left)
        ref_buttons.pack(fill="x", pady=5)
        ttk.Button(ref_buttons, text="从图片框选", command=self._add_reference).pack(side="left", fill="x", expand=True)
        ttk.Button(ref_buttons, text="从视频取帧框选", command=self._add_reference_from_video).pack(side="left", fill="x", expand=True, padx=(6, 0))
        ttk.Button(ref_buttons, text="清空", command=self._clear_references).pack(side="left", padx=(6, 0))
        self.ref_list = tk.Listbox(left, height=5, bg="#111e32", fg="#dbeafe", selectbackground="#31588a", borderwidth=0)
        self.ref_list.pack(fill="x", pady=(4, 5))
        ttk.Button(left, text="保存/更新到实时监控目标库", command=self._save_profile).pack(fill="x", pady=(0, 5))
        self.profile_list = tk.Listbox(left, height=4, bg="#0b1728", fg="#c9dbef", selectbackground="#31588a", borderwidth=0)
        self.profile_list.pack(fill="x", pady=(0, 5))
        profile_actions = ttk.Frame(left)
        profile_actions.pack(fill="x", pady=(0, 9))
        ttk.Button(profile_actions, text="载入选中目标", command=self._load_selected_profile).pack(side="left", fill="x", expand=True)
        ttk.Button(profile_actions, text="启用/停用", command=self._toggle_selected_profile).pack(side="left", padx=(6, 0))
        ttk.Button(profile_actions, text="删除", command=self._delete_selected_profile).pack(side="left", padx=(6, 0))

        ttk.Separator(left).pack(fill="x", pady=4)
        ttk.Label(left, text="2. 选择待检索视频", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", pady=(8, 0))
        self.video_var = tk.StringVar()
        video_row = ttk.Frame(left)
        video_row.pack(fill="x", pady=(8, 4))
        ttk.Entry(video_row, textvariable=self.video_var).pack(side="left", fill="x", expand=True)
        ttk.Button(video_row, text="浏览", command=self._choose_video).pack(side="left", padx=(6, 0))

        ttk.Label(left, text="3. 匹配参数", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", pady=(14, 5))
        param = ttk.Frame(left)
        param.pack(fill="x")
        ttk.Label(param, text="灵敏度", width=9).grid(row=0, column=0, sticky="w", pady=4)
        self.sensitivity_var = tk.StringVar(value="标准")
        ttk.Combobox(param, textvariable=self.sensitivity_var, values=("严格", "标准", "宽松"), state="readonly", width=12).grid(row=0, column=1, sticky="ew")
        ttk.Label(param, text="人员模型", width=9).grid(row=1, column=0, sticky="w", pady=4)
        self.person_model_var = tk.StringVar(value=PERSON_MODEL_FUSION)
        ttk.Combobox(param, textvariable=self.person_model_var, values=PERSON_MODEL_OPTIONS, state="readonly", width=28).grid(row=1, column=1, sticky="ew")
        ttk.Label(param, text="抽帧/秒", width=9).grid(row=2, column=0, sticky="w", pady=4)
        self.sample_fps_var = tk.DoubleVar(value=4.0)
        ttk.Spinbox(param, textvariable=self.sample_fps_var, from_=0.5, to=10.0, increment=0.5, width=12).grid(row=2, column=1, sticky="ew")
        self.high_accuracy_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param, text="4K画面启用分块检测（准确优先）", variable=self.high_accuracy_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 2))
        param.columnconfigure(1, weight=1)

        ttk.Label(left, text="4. 结果目录", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", pady=(14, 5))
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT))
        output_row = ttk.Frame(left)
        output_row.pack(fill="x")
        ttk.Entry(output_row, textvariable=self.output_var).pack(side="left", fill="x", expand=True)
        ttk.Button(output_row, text="选择", command=self._choose_output).pack(side="left", padx=(6, 0))

        action = ttk.Frame(left)
        action.pack(fill="x", pady=(18, 8))
        self.start_button = ttk.Button(action, text="开始完整视频查找", style="Primary.TButton", command=self._start)
        self.start_button.pack(side="left", fill="x", expand=True)
        self.stop_button = ttk.Button(action, text="停止", style="Danger.TButton", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(left, text="打开最近结果文件夹", command=self._open_result).pack(fill="x", pady=(0, 8))
        ttk.Label(left, text="车辆验收仍保留；人员须框选同一个具体人。实时监控只使用目标库中已保存并启用的目标。", style="Hint.TLabel", wraplength=380, justify="left").pack(fill="x", pady=(8, 0))

        preview_header = ttk.Frame(right)
        preview_header.pack(fill="x", pady=(0, 7))
        ttk.Label(preview_header, text="视频处理画面", font=("Microsoft YaHei UI", 12, "bold")).pack(side="left")
        self.stats_var = tk.StringVar(value="等待开始")
        ttk.Label(preview_header, textvariable=self.stats_var, style="Hint.TLabel").pack(side="right")
        self.preview = tk.Label(right, bg="#050b15", text="添加参考图并框选目标，然后选择待检索视频", fg="#71839e", font=("Microsoft YaHei UI", 13))
        self.preview.pack(fill="both", expand=True)
        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(right, variable=self.progress_var, maximum=100).pack(fill="x", pady=(8, 8))

        lower = ttk.Panedwindow(right, orient="horizontal")
        lower.pack(fill="x")
        event_frame = ttk.Frame(lower)
        log_frame = ttk.Frame(lower)
        lower.add(event_frame, weight=3)
        lower.add(log_frame, weight=2)
        ttk.Label(event_frame, text="出现事件").pack(anchor="w")
        columns = ("id", "start", "end", "hits", "score")
        self.event_tree = ttk.Treeview(event_frame, columns=columns, show="headings", height=7)
        for key, label, width in (("id", "事件", 55), ("start", "开始", 90), ("end", "结束", 90), ("hits", "命中帧", 70), ("score", "最高相似度", 100)):
            self.event_tree.heading(key, text=label)
            self.event_tree.column(key, width=width, anchor="center")
        self.event_tree.pack(fill="both", expand=True, padx=(0, 8))
        ttk.Label(log_frame, text="运行日志").pack(anchor="w")
        self.log_text = tk.Text(log_frame, height=8, bg="#081321", fg="#b8c9df", insertbackground="white", borderwidth=0, wrap="word", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)
        self._build_live_ui(live_tab)
        self._refresh_profile_list()

    def _build_live_ui(self, parent: ttk.Frame) -> None:
        controls = ttk.Frame(parent, style="Card.TFrame", padding=10)
        controls.pack(fill="x", padx=2, pady=2)
        ttk.Label(controls, text="四路视频源（RTSP / HTTP / 本地摄像头序号 / 视频文件测试）", font=("Microsoft YaHei UI", 12, "bold")).grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 7))
        self.stream_vars: list[dict] = []
        for index in range(4):
            enabled = tk.BooleanVar(value=index == 0)
            name = tk.StringVar(value=f"视频流{index + 1}")
            source = tk.StringVar()
            status = tk.StringVar(value="未启动")
            self.stream_vars.append({"enabled": enabled, "name": name, "source": source, "status": status})
            ttk.Checkbutton(controls, variable=enabled).grid(row=index + 1, column=0, padx=(0, 4))
            ttk.Entry(controls, textvariable=name, width=12).grid(row=index + 1, column=1, padx=3, pady=3, sticky="ew")
            ttk.Entry(controls, textvariable=source).grid(row=index + 1, column=2, padx=3, pady=3, sticky="ew")
            ttk.Button(controls, text="选测试视频", command=lambda i=index: self._choose_stream_file(i)).grid(row=index + 1, column=3, padx=3)
            ttk.Label(controls, textvariable=status, width=17, style="Hint.TLabel").grid(row=index + 1, column=4, padx=4, sticky="w")
        controls.columnconfigure(2, weight=1)
        action = ttk.Frame(controls)
        action.grid(row=1, column=5, rowspan=4, padx=(12, 0), sticky="ns")
        self.live_start_button = ttk.Button(action, text="启动四路实时监控", style="Primary.TButton", command=self._start_live)
        self.live_start_button.pack(fill="x", pady=(0, 6))
        self.live_stop_button = ttk.Button(action, text="停止实时监控", style="Danger.TButton", state="disabled", command=self._stop_live)
        self.live_stop_button.pack(fill="x", pady=(0, 6))
        ttk.Button(action, text="打开结果目录", command=self._open_result).pack(fill="x")
        ttk.Label(action, text="2/3帧确认\n同路持续只报一次\n离开重进立即再报", style="Hint.TLabel", justify="left").pack(anchor="w", pady=(12, 0))

        content = ttk.Panedwindow(parent, orient="horizontal")
        content.pack(fill="both", expand=True, padx=2, pady=(5, 2))
        preview_frame = ttk.Frame(content, style="Card.TFrame", padding=6)
        alert_frame = ttk.Frame(content, style="Card.TFrame", padding=8)
        content.add(preview_frame, weight=4)
        content.add(alert_frame, weight=2)
        self.live_labels: dict[int, tk.Label] = {}
        for index in range(4):
            cell = ttk.Frame(preview_frame)
            cell.grid(row=index // 2, column=index % 2, padx=3, pady=3, sticky="nsew")
            label = tk.Label(cell, bg="#050b15", fg="#71839e", text=f"视频流{index + 1}\n等待启动", font=("Microsoft YaHei UI", 12))
            label.pack(fill="both", expand=True)
            self.live_labels[index] = label
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)

        ttk.Label(alert_frame, text="跨视频流身份状态", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        global_columns = ("global_id", "target", "stream", "time", "score")
        self.global_tree = ttk.Treeview(alert_frame, columns=global_columns, show="headings", height=5)
        for key, label, width in (("global_id", "全局ID", 125), ("target", "登记目标", 100), ("stream", "当前位置", 90), ("time", "最近", 65), ("score", "相似度", 65)):
            self.global_tree.heading(key, text=label)
            self.global_tree.column(key, width=width, anchor="center")
        self.global_tree.pack(fill="x", pady=(0, 10))

        ttk.Label(alert_frame, text="实时报警记录", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        columns = ("time", "global_id", "stream", "target", "transition", "score")
        self.alert_tree = ttk.Treeview(alert_frame, columns=columns, show="headings", height=9)
        for key, label, width in (("time", "时间", 65), ("global_id", "全局ID", 115), ("stream", "视频流", 80), ("target", "目标", 90), ("transition", "跨流轨迹", 130), ("score", "相似度", 65)):
            self.alert_tree.heading(key, text=label)
            self.alert_tree.column(key, width=width, anchor="center")
        self.alert_tree.pack(fill="both", expand=True)
        ttk.Label(alert_frame, text="报警自动保存：高亮截图、目标切片、事件JSON、前10秒+后20秒录像。文件永久保留，由工作人员手动删除。", style="Hint.TLabel", wraplength=410, justify="left").pack(fill="x", pady=(8, 0))

    def _load_config(self) -> None:
        if not CONFIG_PATH.exists():
            return
        try:
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            self.output_var.set(config.get("output_root", str(DEFAULT_OUTPUT)))
            self.category_var.set(config.get("category", "车辆"))
            self.sensitivity_var.set(config.get("sensitivity", "标准"))
            self.person_model_var.set(config.get("person_model", PERSON_MODEL_FUSION))
            saved_fps = float(config.get("sample_fps", 4.0))
            self.sample_fps_var.set(max(4.0, saved_fps) if int(config.get("config_version", 1)) < 2 else saved_fps)
            for item, saved in zip(self.stream_vars, config.get("streams", [])):
                item["enabled"].set(bool(saved.get("enabled", False)))
                item["name"].set(saved.get("name", item["name"].get()))
                item["source"].set(saved.get("source", ""))
        except Exception:
            pass

    def _save_config(self) -> None:
        CONFIG_PATH.write_text(
            json.dumps(
                {
                    "config_version": 2,
                    "output_root": self.output_var.get(),
                    "category": self.category_var.get(),
                    "sensitivity": self.sensitivity_var.get(),
                    "person_model": self.person_model_var.get(),
                    "sample_fps": self.sample_fps_var.get(),
                    "streams": [
                        {
                            "enabled": item["enabled"].get(),
                            "name": item["name"].get(),
                            "source": item["source"].get(),
                        }
                        for item in self.stream_vars
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _add_reference(self) -> None:
        path = filedialog.askopenfilename(
            title=f"选择包含同一个具体{self.category_var.get()}的参考图",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")],
        )
        if not path:
            return
        dialog = CropDialog(self, Path(path), self.category_var.get())
        self.wait_window(dialog)
        if dialog.result is None:
            return
        self.references.append(dialog.result)
        self.ref_list.insert("end", f"参考图 {len(self.references)}：{Path(path).name}（已框选）")
        self._show_reference(dialog.result)
        self._log(f"已加入参考目标：{Path(path).name}")

    def _show_reference(self, crop) -> None:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((900, 520), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")

    def _add_reference_from_video(self) -> None:
        video_text = self.video_var.get().strip()
        if not Path(video_text).is_file():
            video_text = filedialog.askopenfilename(
                title="选择用于截取参考帧的视频",
                filetypes=[("常见视频", "*.mp4 *.avi *.mov *.mkv *.m4v *.ts"), ("所有文件", "*.*")],
            )
            if not video_text:
                return
            self.video_var.set(video_text)
        cap = cv2.VideoCapture(video_text)
        if not cap.isOpened():
            messagebox.showerror("无法打开", "无法读取该视频。")
            return
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count else 0.0
        timestamp = simpledialog.askfloat(
            "输入取帧时间",
            f"请输入视频秒数（0 ～ {duration:.2f}）。\n同一人员建议分别取正面、侧面、背面等至少3帧。",
            parent=self,
            minvalue=0.0,
            maxvalue=max(0.0, duration),
            initialvalue=0.0,
        )
        if timestamp is None:
            cap.release()
            return
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(timestamp * fps))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            messagebox.showerror("取帧失败", f"无法读取 {timestamp:.2f} 秒画面。")
            return
        dialog = CropDialog(self, frame, self.category_var.get())
        self.wait_window(dialog)
        if dialog.result is None:
            return
        self.references.append(dialog.result)
        self.ref_list.insert("end", f"参考图 {len(self.references)}：视频 {timestamp:.2f}s（已框选）")
        self._show_reference(dialog.result)
        self._log(f"已从视频 {timestamp:.2f}s 加入参考目标。")

    def _clear_references(self) -> None:
        self.references.clear()
        self.current_profile_id = None
        self.ref_list.delete(0, "end")
        self.preview.configure(image="", text="参考目标已清空")
        self.preview_photo = None

    def _save_profile(self) -> None:
        name = self.target_var.get().strip()
        if not name or not self.references:
            messagebox.showwarning("无法保存", "请填写目标名称，并至少框选一张参考图。")
            return
        profiles = self.target_library.list_profiles()
        if self.current_profile_id is None and len(profiles) >= 10:
            messagebox.showwarning("目标数量已满", "第一版最多同时保存10个监控目标；请先删除不再使用的目标。")
            return
        try:
            profile = self.target_library.save(
                name,
                self.category_var.get(),
                self.sensitivity_var.get(),
                [image.copy() for image in self.references],
                self.current_profile_id,
                self.person_model_var.get(),
            )
            self.current_profile_id = profile.profile_id
            self._refresh_profile_list()
            self._log(f"目标库已保存：{profile.name}（{profile.category}/{profile.sensitivity}/{profile.person_model}）")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))

    def _refresh_profile_list(self) -> None:
        self.profile_list.delete(0, "end")
        self.profile_rows = self.target_library.list_profiles()
        for profile in self.profile_rows:
            status = "启用" if profile.enabled else "停用"
            model = profile.person_model if profile.category == "人员" else "车辆固定模型"
            self.profile_list.insert("end", f"[{profile.category}] {profile.name} · {profile.sensitivity} · {model} · {status}")

    def _selected_profile(self) -> TargetProfile | None:
        selection = self.profile_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        return self.profile_rows[index] if index < len(self.profile_rows) else None

    def _load_selected_profile(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            messagebox.showinfo("选择目标", "请先在目标库列表中选择一个目标。")
            return
        try:
            self.references = self.target_library.load_references(profile)
            self.current_profile_id = profile.profile_id
            self.target_var.set(profile.name)
            self.category_var.set(profile.category)
            self.sensitivity_var.set(profile.sensitivity)
            self.person_model_var.set(profile.person_model if profile.category == "人员" else PERSON_MODEL_FUSION)
            self.ref_list.delete(0, "end")
            for index in range(len(self.references)):
                self.ref_list.insert("end", f"参考图 {index + 1}：目标库已载入")
            if self.references:
                self._show_reference(self.references[0])
            self._log(f"已载入目标：{profile.name}")
        except Exception as exc:
            messagebox.showerror("载入失败", str(exc))

    def _delete_selected_profile(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            return
        if not messagebox.askyesno("删除目标", f"确认删除目标库中的“{profile.name}”及其参考/正负样本？"):
            return
        try:
            self.target_library.delete(profile.profile_id)
            if self.current_profile_id == profile.profile_id:
                self._clear_references()
            self._refresh_profile_list()
        except Exception as exc:
            messagebox.showerror("删除失败", str(exc))

    def _toggle_selected_profile(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            messagebox.showinfo("选择目标", "请先在目标库列表中选择一个目标。")
            return
        if self.live_running:
            messagebox.showinfo("实时监控运行中", "请先停止实时监控，再修改目标启用状态。")
            return
        try:
            updated = self.target_library.set_enabled(profile.profile_id, not profile.enabled)
            self._refresh_profile_list()
            self._log(f"目标“{updated.name}”已{'启用' if updated.enabled else '停用'}。")
        except Exception as exc:
            messagebox.showerror("修改失败", str(exc))

    def _choose_stream_file(self, index: int) -> None:
        path = filedialog.askopenfilename(
            title=f"选择视频流{index + 1}的测试视频",
            filetypes=[("常见视频", "*.mp4 *.avi *.mov *.mkv *.m4v *.ts"), ("所有文件", "*.*")],
        )
        if path:
            self.stream_vars[index]["source"].set(path)
            self.stream_vars[index]["enabled"].set(True)

    def _start_live(self) -> None:
        if self.live_running:
            return
        self._save_config()
        profiles = [item for item in self.target_library.list_profiles() if item.enabled]
        if not profiles:
            messagebox.showwarning("没有监控目标", "请先在离线页框选车辆或人员，并保存到实时监控目标库。")
            return
        streams = [
            StreamConfig(index, item["name"].get().strip() or f"视频流{index + 1}", item["source"].get().strip(), item["enabled"].get())
            for index, item in enumerate(self.stream_vars)
        ]
        if not any(item.enabled and item.source for item in streams):
            messagebox.showwarning("没有视频源", "请至少启用一路并填写RTSP/HTTP地址、摄像头序号或测试视频文件。")
            return
        # 实时任务优先；若离线任务正在执行，先安全停止并等待结果落盘。
        if self.worker is not None and self.worker.is_alive():
            self.stop_event.set()
            self._log("实时监控请求已接收：正在停止离线任务，完成保存后自动启动实时监控。")
        self.live_start_button.configure(state="disabled")
        self.live_stop_button.configure(state="normal")

        def work() -> None:
            try:
                while self.worker is not None and self.worker.is_alive():
                    time.sleep(0.1)
                if self.pipeline is None:
                    self.pipeline = InstanceSearchPipeline(PROJECT_ROOT, log=lambda message: self.messages.put({"type": "log", "message": message}))
                    self.messages.put({"type": "device", "text": "本地推理 · 四路自适应抽帧（阈值不降低）"})
                manager = LiveMonitorManager(
                    self.pipeline,
                    self.target_library,
                    profiles[:10],
                    streams,
                    Path(self.output_var.get().strip() or DEFAULT_OUTPUT),
                    self.messages.put,
                )
                self.live_manager = manager
                manager.start()
                self.messages.put({"type": "live_started"})
            except Exception as exc:
                self.messages.put({"type": "live_error", "message": str(exc), "traceback": traceback.format_exc()})

        threading.Thread(target=work, daemon=True, name="start-live").start()

    def _stop_live(self) -> None:
        if self.live_manager is not None:
            self.live_manager.stop()
        self.live_stop_button.configure(state="disabled")
        self._log("正在停止实时监控；已触发的后20秒录像会继续写入完成。")

    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="选择待检索视频",
            filetypes=[("常见视频", "*.mp4 *.avi *.mov *.mkv *.m4v *.ts"), ("所有文件", "*.*")],
        )
        if path:
            self.video_var.set(path)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择结果根目录", initialdir=self.output_var.get() or str(PROJECT_ROOT))
        if path:
            self.output_var.set(path)

    def _validate(self) -> SearchConfig | None:
        if not self.references:
            messagebox.showwarning("缺少参考目标", f"请先添加参考图并框选同一个具体{self.category_var.get()}。")
            return None
        video = Path(self.video_var.get().strip())
        if not video.is_file():
            messagebox.showwarning("缺少视频", "请选择有效的待检索视频文件。")
            return None
        target_name = self.target_var.get().strip()
        if not target_name:
            messagebox.showwarning("缺少名称", "请填写目标名称。")
            return None
        output = Path(self.output_var.get().strip() or DEFAULT_OUTPUT)
        try:
            sample_fps = float(self.sample_fps_var.get())
        except (ValueError, tk.TclError):
            sample_fps = 4.0
        return SearchConfig(
            target_name=target_name,
            video_path=video,
            output_root=output,
            category=self.category_var.get(),
            person_model=self.person_model_var.get(),
            sensitivity=self.sensitivity_var.get(),
            sample_fps=max(0.5, min(10.0, sample_fps)),
            high_accuracy=self.high_accuracy_var.get(),
        )

    def _start(self) -> None:
        config = self._validate()
        if config is None:
            return
        self._save_config()
        references = [image.copy() for image in self.references]
        if self.live_running or self.live_manager is not None:
            self.pending_offline = (config, references)
            self._log("实时监控优先：离线视频任务已加入队列，停止实时监控后自动开始。")
            messagebox.showinfo("已加入队列", "实时监控运行期间不抢占GPU；该离线任务已排队，停止实时监控后自动开始。")
            return
        self._launch_offline(config, references)

    def _launch_offline(self, config: SearchConfig, references: list) -> None:
        self.stop_event = threading.Event()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress_var.set(0)
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)
        self._log("开始任务。首次运行加载模型需要一些时间。")
        def work() -> None:
            try:
                if self.pipeline is None:
                    self.pipeline = InstanceSearchPipeline(PROJECT_ROOT, log=lambda message: self.messages.put({"type": "log", "message": message}))
                    self.messages.put({"type": "device", "text": "本地推理 · YOLOv8l + 车辆/人员FastReID + DINOv2"})
                self.pipeline.search_video(config, references, self.stop_event, progress=self.messages.put)
            except Exception as exc:
                self.messages.put({"type": "error", "message": str(exc), "traceback": traceback.format_exc()})

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self._log("正在停止；当前已完成结果仍会保存……")
        self.stop_button.configure(state="disabled")

    def _poll_messages(self) -> None:
        try:
            while True:
                message = self.messages.get_nowait()
                kind = message.get("type")
                if kind == "log":
                    self._log(message["message"])
                elif kind == "device":
                    self.device_label.configure(text=message["text"])
                elif kind == "progress":
                    self._handle_progress(message)
                elif kind == "done":
                    self._handle_done(message)
                elif kind == "live_started":
                    self.live_running = True
                    self.live_start_button.configure(state="disabled")
                    self.live_stop_button.configure(state="normal")
                    self._log("四路实时监控推理线程已就绪。")
                elif kind == "live_status":
                    index = int(message["stream"])
                    if 0 <= index < len(self.stream_vars):
                        detail = message.get("detail", "")
                        self.stream_vars[index]["status"].set(message["status"] + (f" · {detail}" if detail else ""))
                elif kind == "live_preview":
                    self._handle_live_preview(message)
                elif kind == "alert":
                    self._handle_alert(message)
                elif kind == "global_identity":
                    self._handle_global_identity(message)
                elif kind == "recording_done":
                    self._log(f"报警录像已保存：{message['path']}（{message['frames']}帧）")
                elif kind == "live_stopped":
                    self.live_running = False
                    self.live_manager = None
                    self.live_start_button.configure(state="normal")
                    self.live_stop_button.configure(state="disabled")
                    self._log("实时监控已停止。")
                    if self.pending_offline is not None:
                        config, references = self.pending_offline
                        self.pending_offline = None
                        self._log("正在启动队列中的离线视频任务。")
                        self._launch_offline(config, references)
                elif kind == "live_error":
                    self.live_running = False
                    self.live_start_button.configure(state="normal")
                    self.live_stop_button.configure(state="disabled")
                    self._log(message.get("traceback", message["message"]))
                    messagebox.showerror("实时监控失败", message["message"])
                elif kind == "error":
                    self._log(message.get("traceback", message["message"]))
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    messagebox.showerror("运行失败", message["message"])
        except queue.Empty:
            pass
        self.after(80, self._poll_messages)

    def _handle_live_preview(self, message: dict) -> None:
        index = int(message["stream"])
        label = self.live_labels.get(index)
        if label is None:
            return
        frame = message["frame"]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(420, label.winfo_width() - 6)
        max_h = max(230, label.winfo_height() - 6)
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.live_photos[index] = photo
        label.configure(image=photo, text="")

    def _handle_alert(self, message: dict) -> None:
        profile: TargetProfile = message["profile"]
        self.alert_tree.insert(
            "",
            0,
            values=(datetime.now().strftime("%H:%M:%S"), message["global_id"], message["stream_name"], profile.name, message["transition"], f"{message['score']:.3f}"),
        )
        self._log(f"报警：{message['global_id']} · {message['transition']} · {profile.name}，相似度 {message['score']:.3f}，目录 {message['output_dir']}")
        popup = tk.Toplevel(self)
        popup.title(f"目标报警 — {profile.name} / {message['stream_name']}")
        popup.geometry("960x650")
        popup.configure(bg="#07101f")
        ttk.Label(popup, text=f"{message['global_id']} · {message['transition']} · {profile.name}（{profile.category}）  相似度 {message['score']:.3f}", style="Title.TLabel").pack(fill="x", padx=12, pady=10)
        rgb = cv2.cvtColor(message["frame"], cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((920, 520), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        view = tk.Label(popup, image=photo, bg="#050b15")
        view.image = photo
        view.pack(fill="both", expand=True, padx=12)
        bar = ttk.Frame(popup)
        bar.pack(fill="x", padx=12, pady=10)
        ttk.Label(bar, text=f"证据目录：{message['output_dir']}", style="Hint.TLabel").pack(side="left")
        ttk.Button(bar, text="关闭", command=popup.destroy).pack(side="right")

    def _handle_global_identity(self, message: dict) -> None:
        profile_id = message["profile_id"]
        values = (
            message["global_id"],
            message["target"],
            message["stream_name"],
            message["time"],
            f"{message['score']:.3f}",
        )
        row_id = self.global_tree_rows.get(profile_id)
        if row_id and self.global_tree.exists(row_id):
            self.global_tree.item(row_id, values=values)
            self.global_tree.move(row_id, "", 0)
        else:
            self.global_tree_rows[profile_id] = self.global_tree.insert("", 0, values=values)

    def _handle_progress(self, message: dict) -> None:
        self.progress_var.set(message["percent"])
        self.stats_var.set(
            f"{format_time(message['timestamp'])}/{format_time(message['duration'])} · "
            f"抽帧 {message['sampled']} · 候选 {message['detections']} · 命中 {message['matches']} · 事件 {message['events']}"
        )
        frame = message["frame"]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w = max(640, self.preview.winfo_width() - 8)
        max_h = max(400, self.preview.winfo_height() - 8)
        image.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")

    def _handle_done(self, message: dict) -> None:
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.progress_var.set(100 if not message.get("cancelled") else self.progress_var.get())
        self.last_output = Path(message["output_dir"])
        events = message["events"]
        for event in events:
            self.event_tree.insert(
                "",
                "end",
                values=(event.index, format_time(event.start_s), format_time(event.end_s), event.hit_count, f"{event.best_score:.3f}"),
            )
        status = "已停止并保存" if message.get("cancelled") else "处理完成"
        candidate_count = len(message.get("candidate_events", []))
        self.stats_var.set(f"{status} · 确认事件 {len(events)} · 人工候选 {candidate_count} · 耗时 {message['elapsed']:.1f}s")
        self._log(f"{status}：确认事件 {len(events)} 个，人工候选 {candidate_count} 个，结果目录：{self.last_output}")
        messagebox.showinfo(status, f"确认出现事件 {len(events)} 个；人工复核候选 {candidate_count} 个。\n结果已保存到：\n{self.last_output}")

    def _log(self, text: str) -> None:
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")

    def _open_result(self) -> None:
        path = self.last_output or Path(self.output_var.get() or DEFAULT_OUTPUT)
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))
        except OSError as exc:
            messagebox.showerror("无法打开", str(exc))

    def _close(self) -> None:
        self.stop_event.set()
        if self.live_manager is not None:
            self.live_manager.stop()
        self._save_config()
        self.destroy()


if __name__ == "__main__":
    InstanceSearchApp().mainloop()
