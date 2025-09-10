# experiments_Machine-Learning
Machine-Learning 学习过程中的实验代码



## 🧱 一、目录结构说明

### 目录用途说明

| 目录名            | 用途                                      |
| -------------- | --------------------------------------- |
| `docs/`        | Markdown 格式的笔记，如算法原理、API 笔记、调参经验总结等     |
| `notebooks/`   | 学习阶段的 Jupyter 文件，按时间/主题命名，便于回顾          |
| `projects/`    | 模块化的项目代码，例如一个完整的分类模型训练流程                |
| `utils/`       | 可复用的函数，如绘图、指标封装、模型保存与加载                 |
| `datasets/`    | 数据集处理和下载脚本，不直接上传大文件，可以用 `.gitignore` 排除 |
| `experiments/` | 临时测试脚本或概念验证，快速尝试新想法的地方                  |
| `courses/`      | 学习中的课程笔记                  |


### 目录完整结构

```bash
ml-lab/
├── README.md                     # 仓库总说明
├── requirements.txt              # 公共依赖（或分项目使用 environment.yml）
├── docs/                         # 学习文档、总结笔记
│   ├── cheatsheets/
│   └── concept_notes/
├── courses/
├── notebooks/                    # 学习过程中保存的 Jupyter 实验
│   ├── 2024-05-linear-regression.ipynb
│   └── 2024-06-cnn-classification.ipynb
├── projects/                     # 实验型或专题项目（逐渐成熟的代码）
│   ├── regression_basic/
│   │   ├── train.py
│   │   ├── model.py
│   │   ├── data/
│   │   └── README.md
│   ├── cnn_mnist/
│   └── ...
├── utils/                        # 通用工具函数（如 metric 可视化、模型封装）
│   ├── plot_tools.py
│   └── metrics.py
├── datasets/                     # 数据预处理脚本（或数据下载脚本）
│   ├── download_cifar10.py
│   └── prepare_custom_csv.py
└── experiments/                  # 快速实验区（灵感代码、不完整的想法）
    ├── feature_test_20240604.py
    └── baseline_try.ipynb
```

