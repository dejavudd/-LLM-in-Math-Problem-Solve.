# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from transformers import AutoModel, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import os
from tqdm import tqdm
import json
import re
import torch
import sys

initial_prompt="""You are an expert in solving mathematical problems. Please begin by extracting the math problem from the provided image, and then solve it.

Requirements:
	1.	All mathematical formulas and symbols in your response must be written in LaTeX format.
	2.	Organize your response according to the following structure:
	•	Restate the Problem: Clearly and concisely describe the math problem shown in the image.
	•	Solution Approach: Outline your reasoning and the steps taken to solve the problem.
	•	Final Answer: Present the complete solution.
	3. The final answer can only contain the final result number or option.

Strictly follow the format below in your output:
### Think ###
<Restate the problem and outline the solution approach>

### Answer ###
<Final answer>"""

def extract_steps_and_answer(response):
    """
    从模型响应中提取解题步骤和最终答案
    """
    # 尝试多种模式提取答案
    answer = ""
    step = ""
    
    # 方法1: 寻找boxed答案
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    boxed_match = re.search(boxed_pattern, response)
    if boxed_match:
        answer = boxed_match.group(1)
    
    # 方法2: 寻找最终答案段落
    if not answer:
        final_answer_patterns = [
            r"### Answer ###\n(.*?)(?=\n###|$)",
            r"Thus, the.*?answer.*?is.*?([A-D]|[\d\.]+)",
            r"Therefore.*?answer.*?is.*?([A-D]|[\d\.]+)",
            r"答案是([A-D])选项",
            r"答案.*?([A-D])",
        ]
        for pattern in final_answer_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                break
    
    # 方法3: 寻找解题步骤
    step_patterns = [
        r"### Think ###\n(.*?)(?=\n### Answer ###|$)",
        r"\*\*.*?Approach.*?\*\*\n(.*?)(?=\n\*\*|$)",
        r"Solution.*?:\n(.*?)(?=\nTh[eu]s|$)",
    ]
    
    for pattern in step_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            step = match.group(1).strip()
            break
    
    # 如果还没找到步骤，使用整个响应作为步骤
    if not step:
        step = response.strip()
        
    # 如果没找到明确答案，尝试从步骤中提取
    if not answer and step:
        # 在步骤末尾查找答案
        end_answer_patterns = [
            r"答案是([A-D])",
            r"选择([A-D])",
            r"答案.*?([A-D])",
            r"=\s*([A-D]|\d+\.?\d*)",
            r"answer.*?is.*?([A-D]|\d+\.?\d*)",
        ]
        for pattern in end_answer_patterns:
            match = re.search(pattern, step, re.IGNORECASE)
            if match:
                answer = match.group(1)
                break

    return step, answer

def load_model(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    # Load the model on the available device(s) 
    # Qwen2-VL-2B is a lightweight model under 4GB, optimized for mathematical reasoning and OCR
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    # Qwen2-VL uses AutoProcessor instead of separate tokenizer
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_qwen(image_file):
    """Load image for Qwen2-VL processing"""
    image = Image.open(image_file).convert('RGB')
    return image

def load_jsonl(input_file):
    """加载jsonl文件"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def main(image_dir, input_jsonl, output_jsonl, model_path=None):
    # 导入qwen_vl_utils
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        print("Please install qwen-vl-utils: pip install qwen-vl-utils")
        return
    
    # 使用指定的模型路径，如果没有指定则使用默认的轻量化模型
    if model_path and os.path.exists(model_path):
        model, processor = load_model(model_path)
    else:
        # 使用轻量化的Qwen2-VL-2B模型
        model, processor = load_model("Qwen/Qwen2-VL-2B-Instruct")
    
    input_file = load_jsonl(input_jsonl)
    res = []
    
    for obj in tqdm(input_file, desc="Processing"):
        image_path = os.path.join(image_dir, obj['image'])
        
        # Load image for Qwen2-VL
        image = load_image_qwen(image_path)
        
        # Create the messages for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": initial_prompt},
                ],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        response = output_text[0] if output_text else ""
        
        step, answer = extract_steps_and_answer(response)
        obj["step"] = step
        obj["answer"] = answer
        res.append(obj)
    
    # Save the results to a JSONL file
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in res:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')



if __name__ == "__main__":
    # 支持可选的模型路径参数
    if len(sys.argv) >= 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
