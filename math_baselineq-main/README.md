# 🔢 数学题图像求解系统

> 基于 Qwen2-VL-2B 模型的智能数学题图像识别与求解系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.37.2+-green.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 项目简介

这是一个基于 **Qwen2-VL-2B** 模型的智能数学题求解系统，能够自动识别包含数学题的图片，提取题目内容，生成详细的解题思路，并给出最终答案。系统支持多种题型，包括选择题、填空题和计算应用题。

### ✨ 主要特性

- 🎯 **高精度识别**：采用 Qwen2-VL-2B 模型，专门优化的OCR和公式识别能力
- 🧮 **多题型支持**：支持选择题、填空题、计算应用题等多种数学题型
- 📝 **详细解答**：自动生成完整的解题思路和步骤（LaTeX格式）
- 🚀 **轻量化模型**：模型大小约4GB，支持GPU加速推理
- 📊 **批量处理**：支持JSONL格式的批量图片处理
- 🔧 **灵活配置**：支持自定义模型路径和环境配置

### 🎨 模型优势

- **轻量高效**：Qwen2-VL-2B 模型仅约4GB，适合本地部署
- **数学专长**：在数学推理和公式识别方面表现优秀
- **OCR优化**：强化的光学字符识别能力，准确提取图片中的数学公式
- **多模态理解**：结合视觉和文本理解，全面解析数学题目

## 🛠️ 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU推理)
- 内存：至少8GB RAM
- 显存：至少6GB VRAM (GPU推理)

## 📦 安装指南

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/math_baselineq.git
cd math_baselineq
```

### 2. 创建虚拟环境

```bash
# 使用 conda (推荐)
conda create -n math python=3.10 -y
conda activate math

# 或使用 venv
python -m venv math_env
source math_env/bin/activate  # Linux/Mac
# math_env\Scripts\activate    # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**重要提示**：确保安装 `transformers>=4.37.2` 以保证模型正常工作。

### 4. 快速环境搭建（可选）

```bash
bash build_env.sh
```

## 🚀 使用方法

### 数据准备

1. **准备输入文件**：创建JSONL格式的输入文件，每行包含：
```json
{"image": "image_filename.jpg", "tag": "题型标签"}
```

2. **准备图片文件**：将对应的图片文件放在指定目录中

### 示例数据格式

**输入文件 (input.jsonl)**：
```json
{"image": "math_problem_1.jpg", "tag": "选择题"}
{"image": "math_problem_2.png", "tag": "填空题"}
{"image": "math_problem_3.jpeg", "tag": "计算应用题"}
```

**输出文件 (output.jsonl)**：
```json
{
  "image": "math_problem_1.jpg",
  "tag": "选择题", 
  "steps": "详细的解题步骤（LaTeX格式）",
  "answer": "最终答案",
  "step": "简化的解题思路"
}
```

### 运行推理

#### 方法一：使用脚本运行

```bash
bash run.sh
```

#### 方法二：直接运行Python程序

```bash
python run.py <图片目录> <输入JSONL文件> <输出JSONL文件> [可选：模型路径]
```

**示例**：
```bash
python run.py ./images ./input.jsonl ./output.jsonl
```

### 环境变量配置

您可以通过设置环境变量来自定义运行参数：

```bash
export IMAGE_INPUT_DIR="./sample_output/images"  # 图片目录
export QUERY_PATH="./input.jsonl"               # 输入文件路径  
export OUTPUT_PATH="./output.jsonl"             # 输出文件路径
export MODEL_PATH=""                             # 自定义模型路径（可选）
export CONDA_ENV="math"                         # Conda环境名称
```

## 📊 输出格式说明

系统输出的JSONL文件中，每行包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `image` | string | 原始图片文件名 |
| `tag` | string | 题目类型（选择题/填空题/计算应用题） |
| `steps` | string | 详细解题步骤，LaTeX格式的数学公式 |
| `answer` | string | 最终答案 |
| `step` | string | 简化的解题思路 |

## 📁 项目结构

```
math_baselineq/
├── README.md              # 项目说明文档
├── requirements.txt       # Python依赖包列表
├── run.py                # 主程序文件
├── run.sh                # 运行脚本
├── build_env.sh          # 环境构建脚本
├── input.jsonl           # 示例输入文件
├── output.jsonl          # 示例输出文件
└── sample_output/        # 示例数据目录
    ├── images/          # 示例图片
    └── sample_output.jsonl
```

## 🔧 核心功能模块

### 1. 图像预处理
- 支持多种图片格式（JPEG、PNG等）
- 自动RGB转换和尺寸调整
- 动态分辨率处理

### 2. 模型推理
- Qwen2-VL-2B模型加载和初始化
- GPU/CPU自适应推理
- 批量处理优化

### 3. 结果解析
- 智能提取解题步骤和答案
- LaTeX格式数学公式处理
- 多种答案格式识别

## 🎯 支持的题型

- ✅ **选择题**：多选项数学题目
- ✅ **填空题**：需要填入答案的题目  
- ✅ **计算应用题**：复杂的数学计算和应用题
- ✅ **几何题**：包含图形的几何数学题
- ✅ **代数题**：方程、函数等代数题目

## 🚨 注意事项

1. **模型下载**：首次运行时会自动下载Qwen2-VL-2B模型（约4GB）
2. **GPU设置**：默认使用GPU 6，可在代码中修改 `CUDA_VISIBLE_DEVICES`
3. **内存要求**：建议至少8GB RAM用于模型加载
4. **图片质量**：建议使用清晰的数学题图片以获得最佳识别效果

## ⚡ 性能优化建议

- 使用SSD存储以提高I/O性能
- 适当调整batch_size以平衡速度和内存使用
- 对于大批量处理，建议使用GPU加速
- 定期清理临时文件和缓存

## 🐛 常见问题

### Q: 模型加载失败怎么办？
A: 检查网络连接和磁盘空间，确保能够下载Hugging Face模型。

### Q: 识别准确率不高？
A: 确保输入图片清晰，文字和公式可读。可以尝试提高图片分辨率。

### Q: 内存不足错误？
A: 减少批处理大小，或者使用CPU推理模式。

### Q: 输出格式不正确？
A: 检查输入JSONL文件格式，确保每行都是有效的JSON对象。

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进项目！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) - 强大的多模态语言模型
- [Hugging Face Transformers](https://huggingface.co/transformers/) - 优秀的NLP工具库
- [PyTorch](https://pytorch.org/) - 深度学习框架



⭐ 如果这个项目对您有帮助，请给它一个星标！
