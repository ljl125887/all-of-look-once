# 模型准备

仓库不包含权重。请从上游项目按其许可获取可信权重，文件名和网络结构必须匹配；任意同名模型不能保证兼容。

| 位置 | 用途 | 上游 |
|---|---|---|
| `yolov8l.pt` | 车辆/人员检测 | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| `vehicle_instance_search/models/veri_sbs_R50-ibn.pth` | 车辆 FastReID | [FastReID](https://github.com/JDAI-CV/fast-reid) |
| `vehicle_instance_search/models/dinov2_vits14_pretrain.pth` | DINOv2 通用特征 | [DINOv2](https://github.com/facebookresearch/dinov2) |
| `vehicle_instance_search/models/market_sbs_R50-ibn.pth` | 人员 FastReID | [FastReID](https://github.com/JDAI-CV/fast-reid) |
| `vehicle_instance_search/models/osnet_x1_0_msmt17.pth` | 人员 OSNet | [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) |
| `vehicle_instance_search/models/face_detection_yunet_2026may.onnx` | 可选人脸检测 | [OpenCV Zoo](https://github.com/opencv/opencv_zoo) |
| `vehicle_instance_search/models/face_recognition_sface_2021dec.onnx` | 可选人脸特征 | [OpenCV Zoo](https://github.com/opencv/opencv_zoo) |

车辆推理需要前3项。人员推理还需要人员 FastReID 和 OSNet；当前特征提取实现同时计算这两个分支。两个人脸权重全部存在时自动启用人脸辅助。

权重加载依赖其 state_dict 结构。车辆 FastReID 网络采用 ResNet-IBN/NonLocal 结构；请核对代码与上游 checkpoint，不能直接替换成其他 backbone。系统不在启动时自动下载权重，准备完整后可以离线运行。

GPU 使用 CUDA 可用性检测，否则退回 CPU。支持的设备和精度模式仍需在本机验证；大批量高分辨率候选可能占用较多显存。
