import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv("../data/candidates_500.csv")

job_description = """
Looking for a Data Scientist with
Python, SQL, Machine Learning,
AWS and NLP experience.
"""

job_embedding = model.encode([job_description])

scores = []

for _, row in df.iterrows():

    candidate_text = row["resume_text"]

    candidate_embedding = model.encode(
        [candidate_text]
    )

    similarity = cosine_similarity(
        job_embedding,
        candidate_embedding
    )[0][0]

    scores.append(similarity * 100)

df["semantic_score"] = scores

ranked = df.sort_values(
    by="semantic_score",
    ascending=False
)

print(
    ranked[
        [
            "candidate_id",
            "name",
            "semantic_score"
        ]
    ].head(10)
)