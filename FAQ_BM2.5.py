import math
import re
import jieba
def tokenize(text):
    return list(jieba.cut(text))
def extract_keywords(text):
    pattern = r"[\u4e00-\u9fa5]+|[a-zA-Z]+|\d+"
    return re.findall(pattern, text.lower())
FAQ_DB = {
    # =========================
    # Computer Science / Security
    # =========================
    "how does mysql database connection work in python": """
MySQL database connection in Python allows a program to communicate with a relational database server.

It typically works in the following steps:

1. Import a database driver such as pymysql or mysql-connector
2. Establish a TCP connection using host, username, password, and database name
3. Create a cursor object to execute SQL queries
4. Send SQL commands and receive results
5. Close the connection when finished

This mechanism is widely used in backend development and web applications.
""",

    "what is redis and how does it work": """
Redis is an in-memory key-value database designed for extremely fast data access.

Key characteristics:
- Stores data in RAM instead of disk
- Supports data structures like strings, lists, sets, hashes
- Provides persistence options (RDB, AOF)

Common use cases:
- Caching frequently accessed data
- Session storage in web applications
- Real-time counters and ranking systems
""",

    "what is bm25 ranking function in information retrieval": """
BM25 is a ranking function used in search engines to estimate the relevance of documents.

Core ideas:
- Term frequency saturation (repeated words have diminishing impact)
- Inverse document frequency (rare words matter more)
- Document length normalization

BM25 is widely used in systems such as Elasticsearch and modern search engines.
""",

    "what is longest common subsequence algorithm used for": """
The Longest Common Subsequence (LCS) algorithm measures similarity between two sequences.

It finds the longest sequence that appears in both strings in the same order, but not necessarily consecutively.

Applications:
- Text similarity comparison
- Diff tools in version control systems
- Basic natural language matching systems
""",

    # =========================
    # Cryptography / Networking
    # =========================

    "what is bitcoin and how does it work": """
Bitcoin is a decentralized digital currency based on blockchain technology.

Key principles:
- Distributed ledger (blockchain)
- Proof of Work mining mechanism
- Limited supply of 21 million coins

Use cases:
- Digital payments
- Store of value ("digital gold")

Risks:
- High volatility
- Regulatory uncertainty
""",

    "what is monero cryptocurrency used for": """
Monero is a privacy-focused cryptocurrency designed for anonymous transactions.

Privacy technologies:
- Ring signatures (hide sender identity)
- Stealth addresses (hide receiver identity)
- Confidential transactions (hide amounts)

Unlike Bitcoin, Monero is designed to make transactions untraceable.
""",

    "what is gold in economics and physical science": """
Gold is a chemical element (Au) and a traditional store of value in economics.

Physical properties:
- High conductivity
- Resistant to corrosion
- Dense and malleable metal

Economic role:
- Store of value
- Hedge against inflation
- Reserve asset in central banks
""",

    "what is tor network and how does onion routing work": """
Tor (The Onion Router) is an anonymity network that protects user privacy online.

It works using onion routing:
- Data is encrypted in multiple layers
- Each relay removes one layer of encryption
- No single node knows full path

Limitations:
- Reduced network speed
- Not fully immune to traffic analysis
""",

    "what is x25519 key exchange algorithm": """
X25519 is an elliptic-curve Diffie-Hellman key exchange algorithm.

It is used for:
- Secure key exchange over insecure networks
- TLS 1.3 encryption
- Modern secure messaging systems

Advantages:
- High performance
- Strong security guarantees
- Resistant to many classical cryptographic attacks
""",

    # =========================
    # Biology
    # =========================

    "what is dna in biology and what does it do": """
DNA (Deoxyribonucleic Acid) is the genetic material of living organisms.

Functions:
- Stores genetic information
- Controls protein synthesis
- Transmits hereditary traits

Structure:
- Double helix
- Composed of nucleotide bases (A, T, C, G)
""",

    "what is a cell and why is it important in biology": """
A cell is the basic structural and functional unit of life.

Types:
- Prokaryotic cells (bacteria)
- Eukaryotic cells (plants, animals)

Functions:
- Energy production
- Growth and reproduction
- Biological processes regulation
""",

    "what is a virus and how does it infect cells": """
A virus is a microscopic infectious agent that requires a host cell to replicate.

Infection process:
1. Attachment to host cell
2. Entry into cell
3. Replication using host machinery
4. Release of new viral particles

Viruses are not considered fully living organisms.
""",

    # =========================
    # Nutrition
    # =========================

    "what is protein and why is it important for the human body": """
Protein is a macronutrient composed of amino acids.

Functions:
- Builds and repairs tissues
- Produces enzymes and hormones
- Supports immune system

Sources:
- Meat, fish, eggs
- Beans and legumes
- Dairy products
""",

    "what is a vitamin and what role does it play in nutrition": """
Vitamins are essential organic compounds required in small amounts.

Functions:
- Support metabolism
- Maintain immune system
- Enable enzyme activity

Types:
- Water-soluble (B, C)
- Fat-soluble (A, D, E, K)
""",

    "what is calorie in nutrition and energy metabolism": """
A calorie is a unit of energy used to measure food energy.

Role:
- Measures energy intake from food
- Used by the body for metabolism and activity

Energy balance:
- Excess calories → fat storage
- Deficit calories → weight loss
"""
}


docs = []
questions = []

for q in FAQ_DB:
    questions.append(q)
    docs.append(FAQ_DB[q])

total_docs = len(docs)

inverted_index = {}
doc_len = []

i = 0
while i < total_docs:
    tokens = tokenize(docs[i])
    doc_len.append(len(tokens))

    j = 0
    while j < len(tokens):
        term = tokens[j]

        if term not in inverted_index:
            inverted_index[term] = {}

        if i not in inverted_index[term]:
            inverted_index[term][i] = 0

        inverted_index[term][i] += 1
        j += 1

    i += 1


idf = {}
for term in inverted_index:
    df = len(inverted_index[term])
    idf[term] = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)


k1 = 1.5
b = 0.75
avg_doc_len = sum(doc_len) / len(doc_len)


def bm25_score(query_tokens):
    scores = [0.0] * total_docs

    i = 0
    while i < len(query_tokens):
        term = query_tokens[i]

        if term not in inverted_index:
            i += 1
            continue

        posting_list = inverted_index[term]

        for doc_id in posting_list:
            tf = posting_list[doc_id]

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len[doc_id] / avg_doc_len)

            scores[doc_id] += idf[term] * numerator / denominator

        i += 1

    return scores


def match_faq(query):
    best_index = -1
    best_score = 0
    query_tokens = extract_keywords(query)
    i = 0
    while i < len(questions):
        overlap = 0
        j = 0
        while j < len(query_tokens):
            if query_tokens[j] in questions[i]:
                overlap += 1
            j += 1
        if overlap > best_score:
            best_score = overlap
            best_index = i
        i += 1

    return best_index, best_score


def search(query):
    faq_index, faq_score = match_faq(query)

    if faq_score >= 2 and faq_index != -1:
        return "\"\"\"\n" + FAQ_DB[questions[faq_index]] + "\n\"\"\""

    tokens = tokenize(query)
    scores = bm25_score(tokens)

    ranked = []
    i = 0
    while i < len(scores):
        ranked.append((i, scores[i]))
        i += 1

    ranked.sort(key=lambda x: x[1], reverse=True)

    output = "\"\"\"\nBM25 SEARCH RESULTS:\n\n"
    i = 0
    while i < 3:
        doc_id = ranked[i][0]
        output += docs[doc_id] + "\n\n---\n\n"
        i += 1
    output += "\"\"\""
    return output
while True:
    query = input("\nEnter query (exit to quit): ")
    if query == "exit":
        break
    print(search(query))
