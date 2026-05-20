"""
FAQ 客服问答示例：BM25 + LCS 加权匹配

功能说明：
1. 用户输入一个问题。
2. 程序从 FAQ 问法库中找出最相似的一个问题。
3. 返回该问题对应的答案。
4. 即使相似度很低，也强制返回综合得分最高的 top 1 答案。
5. 一个答案可以配置多个相似问法。

算法说明：
1. BM25：
   - 适合做关键词匹配。
   - 用户问题和 FAQ 问法中有相同关键词时，BM25 分数会更高。
   - 比普通 TF-IDF 更适合搜索和问答召回。

2. LCS：
   - Longest Common Subsequence，最长公共子序列。
   - 用来计算两个句子在字符顺序上的相似程度。
   - 适合补充短句、错别字较少、表达接近的 FAQ 问法匹配。

3. 综合得分：
   final_score = bm25_weight * bm25_score + lcs_weight * lcs_score

适用场景：
- FAQ 客服机器人
- 固定问答匹配
- 小型知识库问答
- 不接大模型，只做本地规则匹配的客服系统
"""

import math
import jieba


# =========================
# FAQ 数据
# 一个答案可以对应多个问法
# =========================

faq_items = [
    {
        "questions": [
            "如何申请退款",
            "怎么退款",
            "我要退款",
            "买错了可以退吗",
            "退款流程是什么"
        ],
        "answer": """
您可以在订单详情页点击“申请退款”，填写退款原因后提交申请。
如果订单已经发货，请先确认是否符合售后规则。
"""
    },
    {
        "questions": [
            "订单多久发货",
            "什么时候发货",
            "多久能发货",
            "付款后多久发货",
            "我的订单什么时候寄出"
        ],
        "answer": """
正常情况下，订单会在付款后 24 小时内发货。
如果遇到节假日、活动高峰或库存异常，发货时间可能会延迟。
"""
    },
    {
        "questions": [
            "怎么修改收货地址",
            "地址填错了怎么办",
            "收货地址可以改吗",
            "我想改地址",
            "订单地址错了"
        ],
        "answer": """
如果订单尚未发货，您可以在订单详情页修改收货地址。
如果订单已经发货，建议您联系人工客服协助处理。
"""
    },
    {
        "questions": [
            "忘记密码怎么办",
            "密码忘了",
            "怎么找回密码",
            "登录密码忘记了",
            "无法登录怎么办"
        ],
        "answer": """
您可以在登录页面点击“忘记密码”，然后通过手机号或邮箱重置密码。
如果仍然无法登录，请联系人工客服协助核验账号。
"""
    },
    {
        "questions": [
            "如何联系客服",
            "人工客服在哪里",
            "怎么找人工客服",
            "我要联系人工",
            "在线客服入口在哪里"
        ],
        "answer": """
您可以点击页面右下角的在线客服入口联系我们。
如果当前人工客服繁忙，请留下问题和联系方式，我们会尽快回复。
"""
    }
]


# =========================
# 文本处理函数
# =========================

def cut_words(text):
    """中文分词函数：把一句话切成多个词，用于 BM25 计算。"""

    words = []                          # 保存最终分词结果
    raw_words = jieba.lcut(text)         # jieba 对中文句子进行分词

    i = 0
    while i < len(raw_words):
        word = raw_words[i].strip()      # 去掉词语两边的空白字符

        if word != "":                   # 过滤空字符串
            words.append(word)

        i = i + 1

    return words


def clean_text(text):
    """清洗文本：去掉标点、空格、换行，主要用于 LCS 字符相似度计算。"""

    ignore_chars = " \t\r\n，。！？、,.!?;；:：\"'“”‘’（）()[]【】"   # 不参与 LCS 的字符
    chars = []                                                          # 保存清洗后的字符

    i = 0
    while i < len(text):
        ch = text[i]

        if ch not in ignore_chars:       # 标点和空白不加入计算
            chars.append(ch)

        i = i + 1

    return "".join(chars)


# =========================
# LCS 最长公共子序列
# =========================

def lcs_length(text1, text2):
    """计算两个字符串的最长公共子序列长度。"""

    m = len(text1)                       # 第一个字符串长度
    n = len(text2)                       # 第二个字符串长度

    prev = []                            # 上一行动态规划结果

    i = 0
    while i <= n:
        prev.append(0)                   # 初始化第一行，全是 0
        i = i + 1

    i = 1
    while i <= m:
        curr = []                        # 当前行动态规划结果

        j = 0
        while j <= n:
            curr.append(0)               # 初始化当前行
            j = j + 1

        j = 1
        while j <= n:
            if text1[i - 1] == text2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                if prev[j] >= curr[j - 1]:
                    curr[j] = prev[j]
                else:
                    curr[j] = curr[j - 1]

            j = j + 1

        prev = curr                      # 当前行变成下一轮的上一行
        i = i + 1

    return prev[n]


def lcs_similarity(text1, text2):
    """计算 LCS 相似度，返回 0 到 1 之间的小数。"""

    text1 = clean_text(text1)             # 清洗用户问题
    text2 = clean_text(text2)             # 清洗 FAQ 问法

    if text1 == "" or text2 == "":
        return 0.0

    length = lcs_length(text1, text2)     # 最长公共子序列长度

    if len(text1) >= len(text2):
        base_len = len(text1)             # 用较长字符串长度做分母
    else:
        base_len = len(text2)

    return length / base_len


# =========================
# BM25 相关函数
# =========================

def bm25_idf(word, df, doc_count):
    """计算某个词的 IDF 值。"""

    if word in df:
        word_df = df[word]                # 包含该词的文档数量
    else:
        word_df = 0

    return math.log(1 + (doc_count - word_df + 0.5) / (word_df + 0.5))


def bm25_score(query_words, doc_tf, doc_len, avg_doc_len, df, doc_count):
    """计算用户问题和某一个 FAQ 问法之间的 BM25 分数。"""

    k1 = 1.5                              # BM25 参数：控制词频饱和度，常用 1.2 到 2.0
    b = 0.75                              # BM25 参数：控制文档长度归一化，常用 0.75

    if avg_doc_len <= 0:
        return 0.0

    score = 0.0                           # 当前 FAQ 问法的 BM25 分数

    i = 0
    while i < len(query_words):
        word = query_words[i]             # 用户问题中的一个词

        if word in doc_tf:
            tf = doc_tf[word]             # 该词在当前 FAQ 问法中出现的次数
            idf = bm25_idf(word, df, doc_count)

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)

            score = score + idf * numerator / denominator

        i = i + 1

    return score


def normalize_scores(scores):
    """把 BM25 原始分数归一化到 0 到 1。"""

    max_score = 0.0                       # 保存最大分数

    i = 0
    while i < len(scores):
        if scores[i] > max_score:
            max_score = scores[i]

        i = i + 1

    new_scores = []                       # 保存归一化后的分数

    i = 0
    while i < len(scores):
        if max_score <= 0:
            new_scores.append(0.0)
        else:
            new_scores.append(scores[i] / max_score)

        i = i + 1

    return new_scores


# =========================
# 展开 FAQ 数据
# 多个问法会被展开成多条候选问题
# 但它们仍然对应同一个答案
# =========================

candidate_questions = []                  # 所有候选问法
candidate_answers = []                    # 每个候选问法对应的答案

i = 0
while i < len(faq_items):
    faq = faq_items[i]

    questions = faq["questions"]          # 当前答案对应的多个问法
    answer = faq["answer"]                # 当前答案

    j = 0
    while j < len(questions):
        candidate_questions.append(questions[j])
        candidate_answers.append(answer)

        j = j + 1

    i = i + 1


# =========================
# 构建 BM25 索引
# =========================

doc_tf_list = []                          # 每个 FAQ 问法的词频字典
doc_len_list = []                         # 每个 FAQ 问法的分词长度
df = {}                                   # 每个词出现在多少个 FAQ 问法中

total_doc_len = 0                         # 所有 FAQ 问法的总长度

i = 0
while i < len(candidate_questions):
    question = candidate_questions[i]
    words = cut_words(question)           # 对 FAQ 问法分词

    doc_len_list.append(len(words))       # 保存当前问法长度
    total_doc_len = total_doc_len + len(words)

    tf = {}                               # 当前问法中的词频
    seen = {}                             # 当前问法中出现过的词，用于计算 df

    j = 0
    while j < len(words):
        word = words[j]

        if word in tf:
            tf[word] = tf[word] + 1
        else:
            tf[word] = 1

        seen[word] = 1                    # 同一个词在同一句中只算出现过一次
        j = j + 1

    seen_words = list(seen.keys())        # 转成列表，方便 while 遍历

    j = 0
    while j < len(seen_words):
        word = seen_words[j]

        if word in df:
            df[word] = df[word] + 1
        else:
            df[word] = 1

        j = j + 1

    doc_tf_list.append(tf)                # 保存当前问法词频

    i = i + 1


doc_count = len(candidate_questions)       # FAQ 候选问法总数

if doc_count == 0:
    avg_doc_len = 0
else:
    avg_doc_len = total_doc_len / doc_count


# =========================
# 可调参数
# =========================

bm25_weight = 0.75                         # BM25 权重，关键词匹配越重要，这个值越大
lcs_weight = 0.25                          # LCS 权重，字符顺序相似越重要，这个值越大

show_debug = True                          # True 显示匹配分数；False 只显示客服回答


# =========================
# 主程序
# =========================

print("FAQ 客服机器人已启动")
print("输入 退出 可以结束程序")


while True:
    user_question = input("\n用户：").strip()

    if user_question == "退出":             # 只保留中文“退出”作为退出命令
        print("已退出")
        break

    if user_question == "":
        continue

    query_words = cut_words(user_question)  # 用户问题分词

    bm25_scores = []                        # 保存用户问题与每个候选问法的 BM25 分数

    i = 0
    while i < doc_count:
        score = bm25_score(
            query_words,                    # 用户问题分词结果
            doc_tf_list[i],                 # 当前 FAQ 问法的词频字典
            doc_len_list[i],                # 当前 FAQ 问法长度
            avg_doc_len,                    # FAQ 问法平均长度
            df,                             # 文档频率表
            doc_count                       # FAQ 问法总数
        )

        bm25_scores.append(score)
        i = i + 1

    bm25_norm_scores = normalize_scores(bm25_scores)

    best_index = 0                          # 当前得分最高的 FAQ 下标
    best_score = -1                         # 当前最高综合分
    best_bm25_score = 0                     # 当前最高结果的 BM25 分数
    best_lcs_score = 0                      # 当前最高结果的 LCS 分数

    i = 0
    while i < doc_count:
        faq_question = candidate_questions[i]

        lcs_score = lcs_similarity(
            user_question,                  # 用户输入的问题
            faq_question                    # 当前 FAQ 问法
        )

        final_score = (
            bm25_weight * bm25_norm_scores[i]
            + lcs_weight * lcs_score
        )

        if final_score > best_score:
            best_score = final_score
            best_index = i
            best_bm25_score = bm25_norm_scores[i]
            best_lcs_score = lcs_score

        i = i + 1

    print("\n客服：")
    print(candidate_answers[best_index].strip())

    if show_debug == True:
        print("\n匹配问法：", candidate_questions[best_index])
        print("综合分：", round(best_score, 3))
        print("BM25：", round(best_bm25_score, 3))
        print("LCS：", round(best_lcs_score, 3))
