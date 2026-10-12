# 第三方模型代码说明

项目AGPL-3.0许可适用于本仓库原创应用内容，上游派生实现、依赖和模型权重保留各自许可与署名。

## 检测、特征与依赖

- Ultralytics YOLO：https://github.com/ultralytics/ultralytics 。使用其AGPL-3.0或适用的商业许可，集成发布须遵守所选条款。
- FastReID：https://github.com/JDAI-CV/fast-reid 。推理结构参考其实现，保留上游Apache-2.0许可要求；checkpoint与训练数据的使用条件须分别确认。
  FastReID Copyright (c) JD AI Research。Apache-2.0全文见 `licenses/FastReID-Apache-2.0.txt`。本仓库的 `reid_models.py` 精简并调整了其网络结构和推理加载流程。
- DINOv2：https://github.com/facebookresearch/dinov2 。本项目通过timm运行其模型，依赖及权重遵循上游适用许可。
- PyTorch、torchvision、timm、OpenCV和Pillow等依赖各自保留其许可证。

仓库不分发任何预训练权重。使用者自行取得模型，并核对训练数据、权重和代码各自的使用条件。

## deep-person-reid / OSNet

`person_osnet.py` 的网络结构精简自 Kaiyang Zhou 的
[deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid)，按 MIT License 使用。

Copyright (c) 2018 Kaiyang Zhou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## OpenCV Zoo / YuNet / SFace

人员实时识别在清晰正脸出现时使用OpenCV Zoo的YuNet人脸检测和SFace特征作为
辅助证据。模型与代码来源：https://github.com/opencv/opencv_zoo 。YuNet目录声明
MIT License；SFace目录声明Apache-2.0。预训练权重的训练数据许可仍应在正式商业
部署前单独复核；本程序不把人脸图像或特征发送到外部服务。
