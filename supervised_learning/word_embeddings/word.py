from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

# Sample dataset of common knowledge facts
common_facts = [
    "The sky is blue.",
    "Water boils at 100 degrees Celsius.",
    "Humans need oxygen to survive.",
    "The Earth revolves around the Sun.",
    "Dogs are mammals."
]

# New fact to evaluate
new_fact = "Bananas are berries."

# Preprocess the facts (tokenize and lowercase)
processed_common_facts = [simple_preprocess(fact) for fact in common_facts]
processed_new_fact = simple_preprocess(new_fact)

# Train Word2Vec model on common facts
model = Word2Vec(sentences=processed_common_facts, vector_size=50, window=5, min_count=1, sg=0)

# Function to compute the average vector for a fact
def get_avg_vector(model, words):
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

# Get average vectors for common facts
common_fact_vectors = [get_avg_vector(model, fact) for fact in processed_common_facts]

# Get average vector for the new fact
new_fact_vector = get_avg_vector(model, processed_new_fact)

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

# Compute similarity of the new fact with each common fact
similarities = [cosine_similarity(new_fact_vector, common_fact_vector) for common_fact_vector in common_fact_vectors]

# Print the similarity scores
for i, similarity in enumerate(similarities):
    print(f"Similarity with common fact {i + 1}: {similarity:.4f}")

# Determine if the fact is surprising based on average similarity
avg_similarity = np.mean(similarities)
threshold = 0.3  # You can adjust the threshold based on your preference
if avg_similarity < threshold:
    print(f"The fact '{new_fact}' is likely surprising.")
else:
    print(f"The fact '{new_fact}' is not surprising.")
