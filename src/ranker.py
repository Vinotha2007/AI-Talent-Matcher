import pandas as pd

# Load dataset
candidates = pd.read_csv("../data/candidates_500.csv")

# Job Requirements
required_skills = {
    "Python",
    "SQL",
    "Machine Learning"
}

required_experience = 3

scores = []
reasons = []

for _, row in candidates.iterrows():

    candidate_skills = set(
        skill.strip()
        for skill in row["skills"].split(",")
    )

    matched_skills = len(
        candidate_skills.intersection(required_skills)
    )

    skill_score = (
        matched_skills /
        len(required_skills)
    ) * 100

    exp_score = min(
        row["experience_years"] /
        required_experience,
        1
    ) * 100

    github_score = row["github_score"]
    linkedin_score = row["linkedin_activity"]

    final_score = (
        0.50 * skill_score +
        0.20 * exp_score +
        0.15 * github_score +
        0.15 * linkedin_score
    )

    scores.append(final_score)

    reasons.append(
        f"Matched {matched_skills}/{len(required_skills)} skills, "
        f"Experience: {row['experience_years']} yrs"
    )

candidates["score"] = scores
candidates["reason"] = reasons

ranked = candidates.sort_values(
    by="score",
    ascending=False
)

print("\nTOP 10 CANDIDATES\n")

print(
    ranked[
        [
            "candidate_id",
            "name",
            "score",
            "reason"
        ]
    ].head(10)
)

ranked.to_csv(
    "../data/ranked_candidates.csv",
    index=False
)

print("\nResults exported!")