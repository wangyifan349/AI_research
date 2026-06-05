from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer  # 导入 M2M100 模型和分词器 / Import M2M100 model and tokenizer
from langdetect import detect, DetectorFactory  # 导入语言检测工具 / Import language detection tools

DetectorFactory.seed = 0  # 固定随机种子，让语言检测结果更稳定 / Fix random seed for more stable language detection
MODEL_NAME = "facebook/m2m100_418M"  # 使用 Facebook 的 M2M100 翻译模型 / Use Facebook M2M100 translation model
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)  # 加载分词器 / Load tokenizer
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME)  # 加载翻译模型 / Load translation model

def detect_language(text: str) -> str:  # 定义语言检测函数 / Define language detection function
    lang = detect(text)  # 自动检测输入文本语言 / Detect the language of input text
    return "zh" if lang.startswith("zh") else lang  # 中文统一返回 zh，其他语言直接返回检测结果 / Return zh for Chinese, otherwise return detected language

def translate_text(text: str) -> str:  # 定义翻译函数 / Define translation function
    source_lang = detect_language(text)  # 检测源语言 / Detect source language
    if source_lang not in {"zh", "en"}:  # 只支持中文和英文 / Only support Chinese and English
        return f"暂不支持该语言：{source_lang}"  # 返回不支持提示 / Return unsupported language message
    target_lang = "en" if source_lang == "zh" else "zh"  # 中文翻译成英文，英文翻译成中文 / Chinese to English, English to Chinese
    tokenizer.src_lang = source_lang  # 设置源语言 / Set source language
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)  # 将文本转换成模型输入 / Convert text into model input
    outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang), max_length=512)  # 生成翻译结果 / Generate translation output
    return tokenizer.decode(outputs[0], skip_special_tokens=True)  # 解码模型输出 / Decode model output

while True:  # 无限循环，持续接收用户输入 / Infinite loop to keep receiving user input
    user_input = input(">>> ").strip()  # 读取用户输入并去掉首尾空格 / Read user input and remove spaces
    if user_input.lower() in {"q", "quit", "exit"}:  # 输入 q、quit 或 exit 时退出程序 / Exit when input is q, quit, or exit
        break  # 跳出循环 / Break the loop
    if user_input:  # 如果输入不为空 / If input is not empty
        print(translate_text(user_input))  # 翻译并打印结果 / Translate and print result
