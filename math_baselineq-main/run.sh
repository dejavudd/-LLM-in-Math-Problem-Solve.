#!/bin/bash

# 设置默认参数，如果环境变量未定义则使用默认值
IMAGE_INPUT_DIR=${IMAGE_INPUT_DIR:-"./sample_output"}
QUERY_PATH=${QUERY_PATH:-"./sample_output/sample_output.jsonl"}
OUTPUT_PATH=${OUTPUT_PATH:-"./output.jsonl"}
MODEL_PATH=${MODEL_PATH:-""}

# 自动检测conda环境或使用系统python
if command -v conda &> /dev/null; then
    # 如果conda可用，激活指定环境
    CONDA_ENV=${CONDA_ENV:-"math"}
    if conda info --envs | grep -q "$CONDA_ENV"; then
        echo "Using conda environment: $CONDA_ENV"
        eval "$(conda shell.bash hook)"
        conda activate $CONDA_ENV
        PYTHON_CMD="python"
    else
        echo "Conda environment '$CONDA_ENV' not found, using system python"
        PYTHON_CMD="python"
    fi
else
    echo "Conda not found, using system python"
    PYTHON_CMD="python"
fi

# 运行程序
if [ -n "$MODEL_PATH" ]; then
    $PYTHON_CMD run.py "$IMAGE_INPUT_DIR" "$QUERY_PATH" "$OUTPUT_PATH" "$MODEL_PATH"
else
    $PYTHON_CMD run.py "$IMAGE_INPUT_DIR" "$QUERY_PATH" "$OUTPUT_PATH"
fi
