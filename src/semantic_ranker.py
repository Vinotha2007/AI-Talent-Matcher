from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

job_description = """
Looking for a Data Scientist with
Python, SQL, Machine Learning,
AWS and NLP experience.
"""

candidate_profile = """
Python, SQL, Deep Learning,
TensorFlow, AWS
"""

job_embedding = model.encode(
    [job_description]
)

candidate_embedding = model.encode(
    [candidate_profile]
)

similarity = cosine_similarity(
    job_embedding,
    candidate_embedding
)

print(
    "Match Percentage:",
    round(similarity[0][0] * 100, 2),
    "%"
)