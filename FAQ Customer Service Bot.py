"""
FAQ Customer Service Bot Based on BM25 and LCS
This file implements a lightweight FAQ question-answering system for customer service scenarios. The program receives a user question, compares it with predefined FAQ question variants, and returns the answer connected to the best-matching FAQ entry. Each FAQ answer can have multiple question variants, so different user expressions such as "How do I request a refund?", "I want a refund", and "Can I get my money back?" can all point to the same answer.
The matching logic uses a weighted combination of two algorithms: BM25 and LCS. BM25 is a classic keyword-based ranking algorithm widely used in search engines and information retrieval systems. It gives higher scores when important query words appear in a candidate FAQ question. Compared with simple TF-IDF, BM25 handles term frequency saturation and document length normalization better, making it more suitable for search-style matching. LCS means Longest Common Subsequence. It measures how many characters appear in the same relative order between two strings. In this FAQ bot, LCS is used as a supplementary similarity signal, especially for short questions or questions with similar wording.
The final score is calculated as:
final_score = bm25_weight * normalized_bm25_score + lcs_weight * lcs_score
This program does not use a low-score rejection rule. No matter how low the similarity is, it always returns the top 1 answer with the highest final score. This behavior is useful when the business requirement is to always provide one FAQ answer instead of returning a fallback message. The code does not use classes and does not use list comprehensions, so the logic remains easy to read and modify for beginners or small projects.
"""

import math
import re
# =========================
# FAQ data
# One answer can have multiple question variants.
# =========================
faq_items = [
    {
        "questions": [
            "How do I request a refund",
            "How can I get a refund",
            "I want a refund",
            "Can I return the product",
            "What is the refund process"
        ],
        "answer": """
You can open the order details page and click "Request Refund".
Please fill in the refund reason and submit the request.
If the order has already been shipped, please check whether it meets the after-sales policy.
"""
    },
    {
        "questions": [
            "When will my order be shipped",
            "How long does shipping take",
            "When do you ship the order",
            "How soon will my order be sent",
            "When will my package be dispatched"
        ],
        "answer": """
Under normal circumstances, your order will be shipped within 24 hours after payment.
Shipping may be delayed during holidays, sales campaigns, or inventory exceptions.
"""
    },
    {
        "questions": [
            "How do I change my shipping address",
            "I entered the wrong address",
            "Can I modify my delivery address",
            "I want to change the address",
            "The order address is wrong"
        ],
        "answer": """
If the order has not been shipped yet, you can modify the shipping address on the order details page.
If the order has already been shipped, please contact customer service for further assistance.
"""
    },
    {
        "questions": [
            "What should I do if I forgot my password",
            "I forgot my password",
            "How can I reset my password",
            "I cannot log in",
            "How do I recover my account"
        ],
        "answer": """
You can click "Forgot Password" on the login page and reset your password using your phone number or email address.
If you still cannot log in, please contact customer service for account verification.
"""
    },
    {
        "questions": [
            "How do I contact customer service",
            "Where is the live agent",
            "How can I talk to a human agent",
            "I want to contact support",
            "Where is the online support entrance"
        ],
        "answer": """
You can contact us by clicking the online customer service entrance in the lower-right corner of the page.
If all agents are busy, please leave your question and contact information, and we will reply as soon as possible.
"""
    }
]


# =========================
# Text processing functions
# =========================

def tokenize(text):
    """Split English text into lowercase word tokens for BM25 calculation."""

    tokens = []                                      # Store final tokens.
    raw_tokens = re.findall(r"[a-zA-Z0-9]+", text)   # Extract English words and numbers.

    i = 0
    while i < len(raw_tokens):
        word = raw_tokens[i].lower().strip()         # Lowercase makes matching case-insensitive.

        if word != "":                               # Ignore empty tokens.
            tokens.append(word)

        i = i + 1

    return tokens


def clean_text(text):
    """Clean text for LCS calculation by keeping only letters and numbers."""

    chars = []                                       # Store cleaned characters.
    text = text.lower()                              # Make LCS case-insensitive.

    i = 0
    while i < len(text):
        ch = text[i]

        if ch.isalnum():                             # Ignore spaces and punctuation.
            chars.append(ch)

        i = i + 1

    return "".join(chars)


# =========================
# LCS: Longest Common Subsequence
# =========================

def lcs_length(text1, text2):
    """Calculate the length of the longest common subsequence between two strings."""

    m = len(text1)                                   # Length of the first string.
    n = len(text2)                                   # Length of the second string.

    prev = []                                        # Previous DP row.

    i = 0
    while i <= n:
        prev.append(0)                               # Initialize the first row with zeros.
        i = i + 1

    i = 1
    while i <= m:
        curr = []                                    # Current DP row.

        j = 0
        while j <= n:
            curr.append(0)                           # Initialize the current row.
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

        prev = curr                                  # Move the current row to the previous row.
        i = i + 1

    return prev[n]


def lcs_similarity(text1, text2):
    """Calculate LCS similarity between two strings and return a value from 0 to 1."""

    text1 = clean_text(text1)                         # Clean the user question.
    text2 = clean_text(text2)                         # Clean the FAQ question.

    if text1 == "" or text2 == "":
        return 0.0

    length = lcs_length(text1, text2)                 # Longest common subsequence length.
    if len(text1) >= len(text2):
        base_len = len(text1)                         # Use the longer string length as denominator.
    else:
        base_len = len(text2)
    return length / base_len
# =========================
# BM25 functions
# =========================

def bm25_idf(word, df, doc_count):
    """Calculate the IDF value of a word."""

    if word in df:
        word_df = df[word]                            # Number of FAQ questions containing this word.
    else:
        word_df = 0

    return math.log(1 + (doc_count - word_df + 0.5) / (word_df + 0.5))


def bm25_score(query_words, doc_tf, doc_len, avg_doc_len, df, doc_count):
    """Calculate the BM25 score between the user question and one FAQ question."""

    k1 = 1.5                                          # BM25 parameter: term frequency saturation.
    b = 0.75                                          # BM25 parameter: document length normalization.

    if avg_doc_len <= 0:
        return 0.0

    score = 0.0                                       # BM25 score for the current FAQ question.

    i = 0
    while i < len(query_words):
        word = query_words[i]                         # One word from the user question.

        if word in doc_tf:
            tf = doc_tf[word]                         # Term frequency in the current FAQ question.
            idf = bm25_idf(word, df, doc_count)       # Importance of the word in all FAQ questions.

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)

            score = score + idf * numerator / denominator

        i = i + 1

    return score


def normalize_scores(scores):
    """Normalize BM25 scores to the range from 0 to 1."""

    max_score = 0.0                                   # Store the maximum score.

    i = 0
    while i < len(scores):
        if scores[i] > max_score:
            max_score = scores[i]

        i = i + 1

    new_scores = []                                   # Store normalized scores.

    i = 0
    while i < len(scores):
        if max_score <= 0:
            new_scores.append(0.0)
        else:
            new_scores.append(scores[i] / max_score)

        i = i + 1

    return new_scores


# =========================
# Expand FAQ data
# Multiple question variants are expanded into separate candidate questions.
# These candidate questions can still point to the same answer.
# =========================

candidate_questions = []                              # All candidate FAQ questions.
candidate_answers = []                                # The answer linked to each candidate question.

i = 0
while i < len(faq_items):
    faq = faq_items[i]

    questions = faq["questions"]                      # Question variants for the current answer.
    answer = faq["answer"]                            # Current FAQ answer.

    j = 0
    while j < len(questions):
        candidate_questions.append(questions[j])
        candidate_answers.append(answer)

        j = j + 1

    i = i + 1


# =========================
# Build BM25 index
# =========================

doc_tf_list = []                                      # Term frequency dictionary for each FAQ question.
doc_len_list = []                                     # Token length of each FAQ question.
df = {}                                               # Document frequency of each word.

total_doc_len = 0                                     # Total token length of all FAQ questions.

i = 0
while i < len(candidate_questions):
    question = candidate_questions[i]
    words = tokenize(question)                        # Tokenize the FAQ question.

    doc_len_list.append(len(words))                   # Store the length of the current question.
    total_doc_len = total_doc_len + len(words)

    tf = {}                                           # Term frequency for the current FAQ question.
    seen = {}                                         # Words already seen in the current FAQ question.

    j = 0
    while j < len(words):
        word = words[j]

        if word in tf:
            tf[word] = tf[word] + 1
        else:
            tf[word] = 1

        seen[word] = 1                                # Count each word once for document frequency.
        j = j + 1

    seen_words = list(seen.keys())                    # Convert keys to a list for while-loop traversal.

    j = 0
    while j < len(seen_words):
        word = seen_words[j]

        if word in df:
            df[word] = df[word] + 1
        else:
            df[word] = 1

        j = j + 1

    doc_tf_list.append(tf)                            # Store term frequency of the current question.

    i = i + 1


doc_count = len(candidate_questions)                  # Total number of candidate FAQ questions.

if doc_count == 0:
    avg_doc_len = 0
else:
    avg_doc_len = total_doc_len / doc_count


# =========================
# Adjustable parameters
# =========================

bm25_weight = 0.75                                    # Higher value means keyword matching is more important.
lcs_weight = 0.25                                     # Higher value means character-order similarity is more important.

show_debug = True                                     # True shows scores; False only shows the customer service answer.


# =========================
# Main program
# =========================

print("FAQ customer service bot has started.")
print("Type stop to end the program.")


while True:
    user_question = input("\nUser: ").strip()

    if user_question == "stop":                       # Only "stop" ends the program.
        print("Program stopped.")
        break

    if user_question == "":
        continue

    query_words = tokenize(user_question)             # Tokenize the user question.

    bm25_scores = []                                  # Store BM25 scores for all candidate questions.

    i = 0
    while i < doc_count:
        score = bm25_score(
            query_words,                              # Tokens from the user question.
            doc_tf_list[i],                           # Term frequency dictionary of the current FAQ question.
            doc_len_list[i],                          # Token length of the current FAQ question.
            avg_doc_len,                              # Average token length of all FAQ questions.
            df,                                       # Document frequency table.
            doc_count                                 # Total number of candidate FAQ questions.
        )

        bm25_scores.append(score)
        i = i + 1

    bm25_norm_scores = normalize_scores(bm25_scores)

    best_index = 0                                    # Index of the best FAQ question.
    best_score = -1                                   # Highest final score found so far.
    best_bm25_score = 0                               # BM25 score of the best result.
    best_lcs_score = 0                                # LCS score of the best result.

    i = 0
    while i < doc_count:
        faq_question = candidate_questions[i]

        lcs_score = lcs_similarity(
            user_question,                            # User input.
            faq_question                              # Current FAQ question.
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
    print("\nCustomer Service:")
    print(candidate_answers[best_index].strip())
    if show_debug == True:
        print("\nMatched Question:", candidate_questions[best_index])
        print("Final Score:", round(best_score, 3))
        print("BM25:", round(best_bm25_score, 3))
        print("LCS:", round(best_lcs_score, 3))
