import os
from sentence_transformers import SentenceTransformer, util

# Get current Python file name
file_name = os.path.basename(__file__)
print("Current file name:", file_name)
# Available English sentence embedding models
models = [
    "sentence-transformers/all-mpnet-base-v2",
    "mixedbread-ai/mxbai-embed-large-v1",
    "BAAI/bge-large-en-v1.5",
    "thenlper/gte-large",
    "intfloat/e5-large-v2"
]

def compare_sentences(sentence1, sentence2, model_name):
    # Load the selected model
    model = SentenceTransformer(model_name)
    # Convert sentences into embeddings
    embedding1 = model.encode(
        sentence1,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    embedding2 = model.encode(
        sentence2,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    # Calculate cosine similarity
    similarity = util.cos_sim(embedding1, embedding2)
    return similarity.item()

print("Available models:")
index = 0
for model in models:
    print(index, model)
    index = index + 1

choice = int(input("Choose a model number: "))
selected_model = models[choice]
sentence1 = input("Enter the first sentence: ")
sentence2 = input("Enter the second sentence: ")
score = compare_sentences(sentence1, sentence2, selected_model)
print("Selected model:", selected_model)
print("Sentence similarity:", score)
