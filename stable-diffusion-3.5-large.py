# pip install -U diffusers transformers accelerate  # 安装依赖 / Install dependencies

import torch  # 导入 PyTorch / Import PyTorch
from diffusers import DiffusionPipeline  # 导入 Diffusers 推理管道 / Import Diffusers pipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",  # 模型名称 / Model name
    torch_dtype=torch.bfloat16,  # 使用 bfloat16 精度，减少显存占用 / Use bfloat16 to reduce VRAM usage
    device_map="cuda"  # 使用 NVIDIA GPU / Use NVIDIA GPU
)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"  # 文生图提示词 / Text-to-image prompt
image = pipe(prompt).images[0]  # 生成图片并取第一张结果 / Generate image and get the first result
image.save("output.png")  # 保存图片 / Save image
