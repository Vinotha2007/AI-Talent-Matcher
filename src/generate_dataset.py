import pandas as pd
import random

skills_pool = [
    "Python", "SQL", "Machine Learning", "Deep Learning",
    "AWS", "Azure", "Power BI", "Tableau",
    "NLP", "TensorFlow", "PyTorch"
]

domains = [
    "Data Science",
    "AI",
    "Cloud",
    "Analytics"
]

data = []

for i in range(1, 501):
    skills = random.sample(skills_pool, random.randint(3, 6))

    data.append({
    "candidate_id": f"C{i:03}",
    "name": f"Candidate_{i}",
    "education": random.choice(
        ["B.Tech", "M.Tech", "MCA", "B.Sc Data Science"]
    ),
    "experience_years": random.randint(0, 10),
    "skills": ", ".join(skills),
    "certifications": random.choice(
        ["AWS", "Azure", "Google", "None"]
    ),
    "projects_count": random.randint(1, 10),
    "github_score": random.randint(50, 100),
    "linkedin_activity": random.randint(50, 100),
    "domain": random.choice(domains),

    "resume_text":
    f"{', '.join(skills)}. "
    f"{random.randint(1,10)} projects completed. "
    f"Certified in {random.choice(['AWS','Azure','Google'])}. "
    f"Domain: {random.choice(domains)}."
})
    df = pd.DataFrame(data)

print(df.columns.tolist())

df.to_csv(
    "../data/candidates_500.csv",
    index=False
)

print("Dataset created successfully!")