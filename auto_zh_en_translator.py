"""
This project provides a simple Chinese-English translation demo
with automatic language detection.
"""
"""
pip install transformers sentencepiece torch
"""


import re
from transformers import pipeline
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
english_to_chinese_translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-zh"
)
chinese_to_english_translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-zh-en"
)
def contains_chinese(text: str) -> bool:
    """Check whether the input text contains Chinese characters."""
    return CHINESE_PATTERN.search(text) is not None
def translate_text(text: str) -> str:
    """Translate Chinese to English, or English to Chinese."""
    if contains_chinese(text):
        translation_result = chinese_to_english_translator(text)
    else:
        translation_result = english_to_chinese_translator(text)
    translated_text = translation_result[0]["translation_text"]
    return translated_text
while True:
    user_input = input(">>> ").strip()
    translated_text = translate_text(user_input)
    print(translated_text)
