from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

job = "Machine Learning Engineer"

candidate = "Deep Learning Developer"

job_embedding = model.encode([job])

candidate_embedding = model.encode([candidate])

similarity = cosine_similarity(
    job_embedding,
    candidate_embedding
)

print(
    "Similarity:",
    similarity[0][0]
)