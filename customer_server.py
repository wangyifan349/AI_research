"""
This program is a simple rule-based customer service question-answering system. It stores predefined customer questions and answers in a dictionary, where one answer can correspond to multiple similar questions by using a tuple of questions as the dictionary key. When a user enters a question, the program first expands all stored question groups into individual candidate questions, then uses jieba to segment Chinese text for BM25 keyword-based matching. At the same time, it calculates the Longest Common Subsequence similarity between the user input and each stored question to measure character-level similarity. The final matching score is computed by combining the normalized BM25 score and the LCS similarity score with fixed weights, usually 0.5 and 0.5. The question with the highest final score is selected as the best match. If the score is lower than the minimum threshold, the program returns a fallback answer; otherwise, it returns the corresponding predefined answer. The program runs continuously in a while True loop, allowing users to input questions repeatedly and receive the most relevant customer service response.
"""
import math  # Math functions
import re  # Regular expressions
from collections import Counter  # Count word frequency
import jieba  # Chinese word segmentation

def tokenize(text):
    words = jieba.lcut_for_search(text.lower())  # Segment text for search
    result = []  # Store valid tokens
    for word in words:
        word = word.strip()  # Remove leading and trailing spaces
        if not word:
            continue  # Skip empty token
        if re.search(r"[\u4e00-\u9fffA-Za-z0-9]", word):  # Keep Chinese, English, and numbers
            result.append(word)  # Add valid token
    return result  # Return token list

def lcs_similarity(text1, text2):
    def clean_text(text):
        return "".join(
            ch.lower()  # Convert character to lowercase
            for ch in text
            if re.match(r"[\u4e00-\u9fffA-Za-z0-9]", ch)  # Keep Chinese, English, and numbers
        )
    a = clean_text(text1)  # Clean first text
    b = clean_text(text2)  # Clean second text
    if not a or not b:
        return 0.0  # Return 0 if either text is empty
    if len(a) < len(b):
        a, b = b, a  # Keep b as the shorter string to save memory
    dp = [0] * (len(b) + 1)  # Dynamic programming array
    for char_a in a:
        prev = 0  # Store previous diagonal value
        for j, char_b in enumerate(b, start=1):
            temp = dp[j]  # Save current value before updating
            if char_a == char_b:
                dp[j] = prev + 1  # Extend common subsequence
            else:
                dp[j] = max(dp[j], dp[j - 1])  # Keep the best previous result
            prev = temp  # Update previous diagonal value
    lcs_len = dp[-1]  # Longest common subsequence length
    return lcs_len / max(len(a), len(b))  # Normalize score to 0-1


def bm25_scores(query, questions, k1=1.5, b=0.75):
    question_tokens = [tokenize(q) for q in questions]  # Tokenize all stored questions
    query_tokens = tokenize(query)  # Tokenize user query
    total_docs = len(questions)  # Total number of stored questions
    if total_docs == 0:
        return []  # Return empty list if there are no questions
    doc_lengths = [len(tokens) for tokens in question_tokens]  # Length of each question
    avg_doc_length = sum(doc_lengths) / total_docs  # Average question length
    df = Counter()  # Document frequency counter
    for tokens in question_tokens:
        for word in set(tokens):
            df[word] += 1  # Count how many questions contain this word
    scores = []  # Store BM25 scores
    for tokens in question_tokens:
        doc_counter = Counter(tokens)  # Word frequency in current question
        doc_length = len(tokens)  # Current question length
        score = 0.0  # BM25 score for current question
        for word in query_tokens:
            if word not in doc_counter:
                continue  # Skip words not found in current question
            tf = doc_counter[word]  # Term frequency in current question
            word_df = df[word]  # Number of questions containing this word
            idf = math.log(
                1 + (total_docs - word_df + 0.5) / (word_df + 0.5)
            )  # Inverse document frequency
            numerator = tf * (k1 + 1)  # BM25 numerator
            denominator = tf + k1 * (
                1 - b + b * doc_length / avg_doc_length
            )  # BM25 denominator with length normalization
            score += idf * numerator / denominator  # Add word score

        scores.append(score)  # Add current question score

    max_score = max(scores)  # Maximum BM25 score

    if max_score == 0:
        return [0.0 for _ in scores]  # Avoid division by zero

    return [score / max_score for score in scores]  # Normalize scores to 0-1


def find_answer(
    query,
    qa_dict,
    bm25_weight=0.5,
    lcs_weight=0.5,
    min_score=0.35
):
    questions = []  # Store all expanded questions
    answers = []  # Store the answer for each expanded question

    for question_group, answer in qa_dict.items():
        if isinstance(question_group, tuple):
            for question in question_group:
                questions.append(question)  # Add one question from tuple
                answers.append(answer)  # Add the same answer
        else:
            questions.append(question_group)  # Add single question
            answers.append(answer)  # Add single answer

    bm25_result = bm25_scores(query, questions)  # Calculate BM25 scores

    best_question = None  # Best matched question
    best_answer = None  # Best matched answer
    best_score = 0.0  # Best final score
    best_bm25 = 0.0  # Best BM25 score
    best_lcs = 0.0  # Best LCS score

    for index, question in enumerate(questions):
        bm25_score = bm25_result[index]  # Current BM25 score
        lcs_score = lcs_similarity(query, question)  # Current LCS score

        final_score = (
            bm25_weight * bm25_score
            + lcs_weight * lcs_score
        )  # Weighted final score

        if final_score > best_score:
            best_score = final_score  # Update best final score
            best_question = question  # Update best question
            best_answer = answers[index]  # Update best answer
            best_bm25 = bm25_score  # Update best BM25 score
            best_lcs = lcs_score  # Update best LCS score

    if best_score < min_score:
        return {
            "answer": "抱歉，我暂时没有找到合适答案。",
            "matched": False,
            "question": None,
            "score": best_score,
            "bm25": best_bm25,
            "lcs": best_lcs,
        }  # Return fallback answer if score is too low

    return {
        "answer": best_answer,
        "matched": True,
        "question": best_question,
        "score": best_score,
        "bm25": best_bm25,
        "lcs": best_lcs,
    }  # Return matched answer and debug scores


qa_dict = {
    (
        "怎么重置密码",
        "忘记密码怎么办",
        "密码忘了怎么找回",
        "如何找回密码",
    ): '''你可以在登录页点击“忘记密码”，然后根据页面提示进行操作。

操作步骤：
    1. 打开登录页面
    2. 点击“忘记密码”
    3. 输入手机号或邮箱
    4. 完成验证后重置密码''',

    (
        "如何修改手机号",
        "怎么更换手机号",
        "手机号可以改吗",
    ): '''请进入账号设置，选择“手机号管理”，然后按照页面提示修改手机号。

注意：
    如果旧手机号无法接收验证码，请联系人工客服处理。''',

    (
        "手机号收不到验证码怎么办",
        "验证码收不到",
        "短信验证码收不到",
    ): '''请先检查以下情况：

    1. 手机号是否填写正确
    2. 短信是否被拦截
    3. 手机是否欠费或信号较弱
    4. 是否频繁获取验证码

如果仍然无法收到验证码，建议稍后重试或联系人工客服。''',

    (
        "订单怎么退款",
        "如何申请退款",
        "我要退款",
    ): '''请进入订单详情页，点击“申请退款”，然后按照页面提示提交退款申请。

退款申请提交后，系统会根据订单状态进行审核。''',

    (
        "退款多久到账",
        "退款什么时候到账",
        "退款多久能收到",
    ): '''退款到账时间通常为 1 到 7 个工作日。

具体到账时间以支付渠道处理时间为准。''',

    (
        "怎么开发票",
        "如何申请发票",
        "我要开发票",
    ): '''请进入订单详情页，选择“申请发票”，填写发票信息后提交。

发票信息提交后，请耐心等待系统处理。''',

    (
        "会员怎么取消自动续费",
        "怎么关闭自动续费",
        "取消会员续费",
    ): '''请进入会员中心，在自动续费管理中关闭自动续费。
关闭后，当前会员权益仍可使用到有效期结束。''',
}  # Stored question-answer dictionary


while True:
    user_input = input("用户：").strip()  # Read user input
    result = find_answer(
        query=user_input,
        qa_dict=qa_dict,
        bm25_weight=0.5,
        lcs_weight=0.5,
        min_score=0.35
    )  # Find the best answer

    print("客服：", result["answer"])  # Print customer service answer
    print(
        "匹配问题：",
        result["question"],
        "| 最终分数：",
        round(result["score"], 4),
        "| BM25：",
        round(result["bm25"], 4),
        "| LCS：",
        round(result["lcs"], 4)
    )  # Print matching details

    print()  # Print blank line
