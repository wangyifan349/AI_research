# pip install -U diffusers transformers accelerate

import torch  # 导入 PyTorch / Import PyTorch
from diffusers import DiffusionPipeline  # 导入 Diffusers 推理管道 / Import Diffusers pipeline

pipe = DiffusionPipeline.from_pretrained(
    "xinsir/controlnet-union-sdxl-1.0",  # 模型名称 / Model name
    torch_dtype=torch.bfloat16,  # 模型精度 / Model precision
    device_map="cuda"  # 使用 GPU / Use GPU
)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"  # 提示词 / Prompt

image = pipe(
    prompt=prompt,  # 输入提示词 / Input prompt
).images[0]  # 取第一张图片 / Get first image

image.save("output.png")  # 保存图片 / Save image
