def skill_match(candidate_skills, required_skills):

    candidate_set = set(
        skill.strip().lower()
        for skill in candidate_skills.split(",")
    )

    required_set = set(
        skill.strip().lower()
        for skill in required_skills.split(",")
    )

    matched = len(candidate_set & required_set)

    return (matched / len(required_set)) * 100