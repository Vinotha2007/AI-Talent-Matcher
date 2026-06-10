import pandas as pd
from sentence_transformers import SentenceTransformer

df = pd.read_csv("../data/candidates_500.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    df["resume_text"].tolist(),
    show_progress_bar=True
)

import pickle

with open("../data/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("Embeddings saved!")