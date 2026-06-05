from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer  # 导入 M2M100 模型和分词器
from langdetect import detect, DetectorFactory  # 导入语言检测工具

DetectorFactory.seed = 0  # 固定随机种子，让语言检测结果更稳定
MODEL_NAME = "facebook/m2m100_418M"  # 使用 Facebook 的 M2M100 翻译模型
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)  # 加载分词器
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)  # 加载翻译模型

def detect_language(text: str) -> str:  # 定义语言检测函数
    lang = detect(text)  # 自动检测输入文本语言
    return "zh" if lang.startswith("zh") else lang  # 中文统一返回 zh，其他语言直接返回原结果

def translate_text(text: str) -> str:  # 定义翻译函数
    source_lang = detect_language(text)  # 检测源语言
    if source_lang not in {"zh", "en"}:  # 只支持中文和英文
        return f"暂不支持该语言：{source_lang}"  # 返回不支持提示
    target_lang = "en" if source_lang == "zh" else "zh"  # 中文翻译成英文，英文翻译成中文
    tokenizer.src_lang = source_lang  # 设置源语言
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)  # 将文本转换成模型输入
    outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang), max_length=512)  # 生成翻译结果
    return tokenizer.decode(outputs[0], skip_special_tokens=True)  # 解码模型输出，得到最终翻译文本

while True:  # 无限循环，持续接收用户输入
    user_input = input(">>> ").strip()  # 读取用户输入并去掉首尾空格
    if user_input.lower() in {"q", "quit", "exit"}:  # 输入 q、quit 或 exit 时退出程序
        break  # 跳出循环
    if user_input:  # 如果输入不为空
        print(translate_text(user_input))  # 翻译并打印结果
