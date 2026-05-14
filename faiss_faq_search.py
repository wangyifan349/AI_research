# pip install -U sentence-transformers faiss-cpu
"""
This FAQ bot uses semantic search instead of exact keyword matching. Each FAQ question is converted into a dense vector embedding by a multilingual SentenceTransformer model. The embedding represents the meaning of the sentence in a numerical form, so questions with similar meanings can be close to each other even when they use different words or different languages.
Before building the search index, all FAQ questions are encoded and normalized. After normalization, comparing vectors with inner product is equivalent to cosine similarity. The normalized FAQ embeddings are stored in a FAISS IndexFlatIP index, which performs exact inner-product search. When the user enters a query, the query is also encoded and normalized in the same way, then FAISS searches for the FAQ question with the highest similarity score.
The system returns the answer associated with the most similar FAQ question. In this implementation, only the FAQ questions are embedded and searched, while the answers are stored as values in a dictionary and returned after retrieval. The model determines semantic similarity, FAISS provides efficient vector search, and the dictionary maps the matched question back to its final answer.
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# 1. Load the sentence encoder model.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)
# 2. FAQ dictionary.
# The key is the question, which is used for vectorization and retrieval.
# The value is the answer. Triple quotes are used to support line breaks and indentation.
FAQ_DICT = {
    "How do I apply for an invoice?": """You can go to the order details page after the order is completed, click "Apply for Invoice", fill in the invoice title, tax number, and email address, then submit the request.""",
    "How long does contract approval usually take?": """Standard contracts usually take 1 to 3 business days for approval. Non-standard contracts require legal review, so the process may take longer.""",
    "How do I modify enterprise account information?": """Please ask the administrator to go to the enterprise admin dashboard and update the company name, contact person, phone number, and other information in "Account Settings".""",
    "Can I get an invoice?": """Yes. After your order is completed, go to the order details page and submit an invoice request.""",
    "How long does contract approval take?": """Standard contracts usually take 1 to 3 business days. Non-standard contracts may require legal review.""",
    "Can courses be replayed?": """Yes. Replay videos are usually generated after the live course ends. Students can view them on the course page.""",
    "How do I apply to withdraw from a course?": """Please go to the learning center, select the corresponding course, and submit a course withdrawal request. Whether the course can be refunded depends on the course rules and purchase time.""",
    "When will exam results be released?": """Exam results are usually released within 3 to 7 business days after the exam ends. The exact time depends on the course notice.""",
    "Can I watch the course replay?": """Yes. Replay videos are usually available on the course page after the live session ends.""",
    "How do I withdraw from a course?": """Please go to the learning center, select the course, and submit a withdrawal request.""",
    "What should I do when I have a fever?": """It is recommended to measure your temperature first, drink enough water, and rest. If a high fever persists or is accompanied by symptoms such as difficulty breathing or abnormal consciousness, seek medical care promptly.""",
    "What should I do if my physical examination report is abnormal?": """It is recommended to consult a doctor with the complete physical examination report. A professional doctor can evaluate it together with your medical history, symptoms, and test results.""",
    "What should I do if I forget to take my medicine?": """Please refer to the medicine instructions or your doctor's advice. Do not double the dose on your own. If you are unsure, consult a doctor or pharmacist.""",
    "What should I do if I have a fever?": """Measure your temperature, drink enough water, and rest. Seek medical care if symptoms are severe or persistent.""",
    "What should I do if my medical report is abnormal?": """Please consult a doctor with the full report so they can interpret it based on your medical history and symptoms."""
}
# 3. Extract the FAQ question list.
# The indices returned by FAISS are mapped back to questions.
questions = list(FAQ_DICT.keys())
# 4. Preprocess FAQ question text.
# Only questions are vectorized here, not answers.
faq_passages = [f"passage: {question}" for question in questions]
# 5. Encode FAQ questions into vectors.
# After normalization, inner product similarity is equivalent to cosine similarity ranking.
question_embeddings = model.encode(
    faq_passages,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")
# 6. Create a FAISS exact inner product index.
dim = question_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(question_embeddings)
# 7. FAQ retrieval function.
# It always returns the most similar answer with top_k=1, regardless of the similarity score.
def search_faq(query: str, top_k: int = 1):
    query_text = f"query: {query}"
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    scores, indices = index.search(query_embedding, top_k)
    best_score = float(scores[0][0])
    best_idx = int(indices[0][0])
    matched_question = questions[best_idx]
    answer = FAQ_DICT[matched_question]
    return {
        "query": query,
        "matched_question": matched_question,
        "answer": answer,
        "score": best_score
    }
# 8. Continuous command-line chat.
# Enter exit / quit / q to stop the program.
def chat_loop():
    print("FAQ Bot started. Enter exit / quit / q to stop.")
    print("-" * 60)
    while True:
        user_query = input("User: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit", "q"]:
            print("System: Chat ended.")
            break
        result = search_faq(user_query, top_k=1)
        print("System:", result["answer"])
        print(f"Matched question: {result['matched_question']}")
        print(f"Similarity score: {result['score']:.4f}")
        print("-" * 60)
# 9. Program entry point.
# Start the chat loop when this file is run directly.
chat_loop()
