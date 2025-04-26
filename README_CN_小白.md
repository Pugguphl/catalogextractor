# 基于视觉-语言大模型的考古图录信息收集方法

[![arXiv](https://img.shields.io/badge/arXiv-2412.20088-b31b1b.svg)](https://arxiv.org/pdf/2412.20088)

本项目包含使用计算机视觉语言大模型自动提取考古目录信息的代码。该方法处理陶器和其他考古文物的图片，提取结构化数据用于分析和记录。

## 概述

该考古图录信息收集方法利用最先进的视觉-语言模型来：

- 检测并定位图录图片中的文物
- 提取图录元数据
- 生成用于考古研究的结构化输出

技术细节请参考[我们的论文](https://arxiv.org/pdf/2412.20088)。


### **安装：anaconda**

安装细节参考教程[Anaconda安装-超详细版(](https://blog.csdn.net/qq_45281589/article/details/134597810)
如果电脑是linux，请选择Anaconda-1.4.0-Linux-x86.sh
如果电脑是Mac，请选择Anaconda-1.4.0-MacOSX-x86_64.sh
如果电脑是Windows，请选择Anaconda-1.4.0-Windows-x86.exe
---

### **安装：catalogextractor**

1. 克隆项目代码库并进入 `catalogextractor` 目录。

```bash
git clone https://github.com/Pugguphl/catalogextractor.git
cd catalogextractor/
```

2. 使用 Python 3.10 创建一个名为 `catalogextractor` 的 Anaconda 环境。

```bash
conda create -n catalogextractor python=3.10 -y
```

3. 激活刚创建的环境。

```bash
conda activate catalogextractor
```

4. 安装所需的依赖库。

```bash
pip install -r requirements.txt
```

---

### **安装：Grounding Dino**

1. 从 GitHub 克隆 GroundingDINO 项目代码库。

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

2. 切换到 GroundingDINO 文件夹。

```bash
cd GroundingDINO/
```

3. 在当前目录安装所需的依赖库。

```bash
pip install -e .
```

4. 下载预训练模型权重。

```bash
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

5. 如果第 4 步中的下载命令失败，请手动下载模型：
   
   访问 [groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) 并保存到第 4 步中创建的 `weights` 文件夹。

---

### **运行陶器信息提取工具**

1. 返回到 `catalogextractor` 根目录。

```bash
cd ..  # 如果当前在 GroundingDINO 目录
```

2. 使用以下命令运行 `pottery_info_extractor.py` 脚本，并提供所需参数：

```bash
python pottery_info_extractor.py --output_dir /path/to/output/directory --image_dir /path/to/pottery/images/folder --config_gd /path/to/grounding_dino/config --pretrained_weight_gd /path/to/grounding_dino/weights
```

3. 将路径参数替换为实际路径：
    - `--output_dir`：提取信息保存的目录
    - `--image_dir`：包含陶器目录图片的文件夹
    - `--config_gd`：Grounding DINO 模型配置文件的路径
    - `--pretrained_weight_gd`：Grounding DINO 预训练模型权重的路径

4. 示例用法：

```bash
python pottery_info_extractor.py --output_dir ./results --image_dir ./pottery_catalog_images --config_gd GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --pretrained_weight_gd GroundingDINO/weights/groundingdino_swint_ogc.pth
```

## 输出结果说明

程序运行后，将在指定的输出目录中生成结构化的结果文件夹。输出结构如下：

```
输出目录/
├── 图片名1/
│   ├── 发掘单位名1/
│   │   ├── 器物类型1/
│   │   │   ├── 编号1.png
│   │   │   ├── 编号2.png
│   │   │   └── ...
│   │   └── 器物类型2/
│   │       └── ...
│   ├── 发掘单位名2/
│   │   └── ...
│   └── error_list_图片名1.txt
├── 图片名2/
│   └── ...
└── ...
```

### 示例

以下是一个输出目录示例：

```
D:\PycharmProjects\sketch_Seg\vlm_collection\output\
└── IF1 IF2 IH1\
    └── QDIF1\
        └── 敞口折腹钵\
            ├── 001.png
            ├── 002.png
            └── ...
```

### 错误记录

在每个图片名文件夹中，会生成一个名为 `error_list_图片名.txt` 的文件，记录处理过程中可能出现问题的器物编号。您可以根据这些错误信息进行人工检查和调整。

错误文件包含:
- 未能正确识别的器物编号
- 图像模糊或信息不全的器物
- 分类或提取过程中出现异常的器物

检查这些错误文件，以确保数据提取的准确性和完整性。
