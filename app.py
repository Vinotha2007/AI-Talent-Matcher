import streamlit as st
st.set_page_config(
    page_title="AI Talent Matcher",
    page_icon="🎯",
    layout="wide"
)
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #f8fbff, #e6f2ff);
}

h1 {
    color: #0F52BA;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)
import pandas as pd
import pickle
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
def create_report(candidate):

    pdf = SimpleDocTemplate("Hiring_Report.pdf")

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "AI Talent Matcher Hiring Report",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 12))

    content.append(
        Paragraph(
            f"Candidate: {candidate['name']}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Score: {candidate['score']:.2f}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Experience: {candidate['experience_years']} years",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Skills: {candidate['skills']}",
            styles["Normal"]
        )
    )

    pdf.build(content)

    return "Hiring_Report.pdf"

# -------------------------
# PAGE TITLE
# -------------------------

st.title("🎯 AI Talent Matcher")

st.caption(
    "AI-Powered Recruitment & Candidate Ranking Platform"
)

# -------------------------
# LOAD DATA
# -------------------------

df = pd.read_csv("data/candidates_500.csv")

with open("data/embeddings.pkl", "rb") as f:
    candidate_embeddings = pickle.load(f)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -------------------------
# RANKING METHOD
# -------------------------

st.sidebar.header("Recruiter Controls")

ranking_method = st.sidebar.radio(
    "Choose Ranking Method",
    ["Rule-Based", "AI Semantic"]
)

# -------------------------
# JOB DESCRIPTION
# -------------------------

st.header("Job Requirements")

job_description = st.text_area(
    "Paste Job Description",
    """We are looking for a Data Scientist
with Python, SQL and Machine Learning."""
)

# -------------------------
# RESUME UPLOAD
# -------------------------

st.subheader("Upload Candidate Resume")

uploaded_file = st.sidebar.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

resume_text = ""

if uploaded_file:

    pdf = PdfReader(uploaded_file)

    for page in pdf.pages:

        text = page.extract_text()

        if text:
            resume_text += text

    st.subheader("Resume Preview")

    st.text_area(
        "Extracted Resume Text",
        resume_text,
        height=200
    )

    # Resume Match Analysis

    st.subheader("Resume Match Analysis")

    job_embedding = model.encode([job_description])

    resume_embedding = model.encode([resume_text])

    similarity = cosine_similarity(
        job_embedding,
        resume_embedding
    )[0][0]

    match_percentage = round(
        similarity * 100,
        2
    )

    st.metric(
        "Resume Match %",
        f"{match_percentage}%"
    )

    if match_percentage >= 80:
        st.success("Excellent Match")
    elif match_percentage >= 60:
        st.warning("Good Match")
    else:
        st.error("Needs Improvement")

# -------------------------
# EXPERIENCE
# -------------------------

experience_required = st.sidebar.slider(
    "Required Experience (Years)",
    0,
    10,
    3
)
# -------------------------
# SKILL DETECTION
# -------------------------

known_skills = [
    "Python",
    "SQL",
    "Machine Learning",
    "Deep Learning",
    "AWS",
    "Azure",
    "Power BI",
    "Tableau",
    "TensorFlow",
    "PyTorch",
    "NLP"
]

required_skills = {
    skill
    for skill in known_skills
    if skill.lower() in job_description.lower()
}

st.write(
    "Detected Skills:",
    ", ".join(required_skills)
)

st.subheader("🧠Suggested Interview Questions")

question_bank = {
    "Python": "Explain the difference between lists and tuples in Python.",
    "SQL": "What is the difference between INNER JOIN and LEFT JOIN?",
    "Machine Learning": "What is overfitting and how can you prevent it?",
    "Deep Learning": "What is the purpose of activation functions?",
    "AWS": "What is the difference between EC2 and S3?",
    "Azure": "What Azure services have you worked with?",
    "TensorFlow": "How do you build a neural network in TensorFlow?",
    "PyTorch": "What advantages does PyTorch offer for research?",
    "NLP": "What is tokenization in NLP?",
    "Power BI": "How do you create interactive dashboards in Power BI?",
    "Tableau": "What are calculated fields in Tableau?"
}

for skill in required_skills:
    if skill in question_bank:
        st.write(f"❓ {question_bank[skill]}")
# -------------------------
# SCORING
# -------------------------

scores = []
reasons = []

if ranking_method == "AI Semantic":

    job_embedding = model.encode([job_description])

    similarities = cosine_similarity(
        job_embedding,
        candidate_embeddings
    )[0]

    scores = similarities * 100

    reasons = [
        f"Semantic Match: {score:.2f}%"
        for score in scores
    ]

else:

    for _, row in df.iterrows():

        candidate_skills = {
            skill.strip()
            for skill in row["skills"].split(",")
        }

        matched_skills = len(
            candidate_skills.intersection(required_skills)
        )

        skill_score = (
            matched_skills /
            len(required_skills)
        ) * 100 if len(required_skills) > 0 else 0

        exp_score = min(
            row["experience_years"] /
            experience_required,
            1
        ) * 100 if experience_required > 0 else 100

        final_score = (
            0.7 * skill_score +
            0.3 * exp_score
        )

        scores.append(final_score)

        reasons.append(
            f"Matched {matched_skills}/{len(required_skills)} skills | "
            f"Experience: {row['experience_years']} years"
        )

# -------------------------
# RESULTS
# -------------------------

df["score"] = scores
df["reason"] = reasons

ranked = df.sort_values(
    by="score",
    ascending=False
)

# -------------------------
# METRICS
# -------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "👥Total Candidates",
    len(df)
)

col2.metric(
    "🏆Top Score",
    round(ranked["score"].max(), 2)
)

col3.metric(
    "📊 Average Score",
    round(ranked["score"].mean(), 2)
)
col4.metric(
    "🤖Ranking Method",
    ranking_method
)
best = ranked.iloc[0]

st.success(
    f"🏆 Best Candidate: {best['name']} | Score: {best['score']:.2f}"
)
st.info(
    f"""
    📌 Recruiter Summary

    Best Candidate: {best['name']}

    Score: {best['score']:.2f}

    Experience: {best['experience_years']} years

    Skills: {best['skills']}
    """
)
# -------------------------
# CANDIDATE PROFILE
# -------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Dashboard",
        "👤 Candidate Profile",
        "🧠 AI Insights"
    ]
)
with tab2:
 st.subheader("Candidate Details")
 search_name = st.text_input(
    "🔍 Search Candidate Name"
)

if search_name:
    ranked = ranked[
        ranked["name"].str.contains(
            search_name,
            case=False
        )
    ]

candidate = st.selectbox(
    "Select Candidate",
    ranked["candidate_id"]
)

selected = ranked[
    ranked["candidate_id"] == candidate
]

st.subheader("👤Candidate Profile")

st.write(
    "Candidate ID:",
    selected.iloc[0]["candidate_id"]
)

st.write(
    "Name:",
    selected.iloc[0]["name"]
)

st.write(
    "Experience:",
    selected.iloc[0]["experience_years"],
    "years"
)

st.write(
    "Skills:",
    selected.iloc[0]["skills"]
)

st.write(
    "Score:",
    round(selected.iloc[0]["score"], 2)
)

st.write(
    "Reason:",
    selected.iloc[0]["reason"]
)
st.subheader("⭐ Hiring Recommendation")
report_path = create_report(
    selected.iloc[0]
)

with open(report_path, "rb") as file:

    st.download_button(
        "📄 Download Hiring Report",
        file,
        file_name="Hiring_Report.pdf"
    )

candidate_score = selected.iloc[0]["score"]

if candidate_score >= 85:
    st.success("✅ Strongly Recommended")

elif candidate_score >= 70:
    st.warning("🟡 Recommended")

else:
    st.error("❌ Not Recommended")
# -------------------------
# SKILL GAP
# -------------------------

candidate_skills = {
    skill.strip()
    for skill in selected.iloc[0]["skills"].split(",")
}

missing_skills = (
    required_skills -
    candidate_skills
)

st.subheader("🎯Skill Gap Analysis")

if missing_skills:
    st.warning(
        "Missing Skills: " +
        ", ".join(missing_skills)
    )
else:
    st.success(
        "Candidate meets all required skills!"
    )
st.subheader("Recommended Learning Path")

if missing_skills:
    for skill in missing_skills:
        st.write(f"📘 Learn {skill}")
else:
    st.success("No additional skills required")
# -------------------------
# TOP 10
# -------------------------
with tab1:
 st.subheader("🏆 Top 10 Candidates")

st.dataframe(
    ranked[
        [
            "candidate_id",
            "name",
            "score",
            "reason"
        ]
    ].head(10)
)
st.subheader("Candidate Score Distribution")

fig, ax = plt.subplots()

ax.hist(
    ranked["score"],
    bins=10
)

ax.set_xlabel("Score")
ax.set_ylabel("Candidates")
ax.set_title("Score Distribution")

st.pyplot(fig)

# -------------------------
# DOWNLOAD
# -------------------------

csv = ranked.to_csv(index=False)

st.download_button(
    "Download Ranked Candidates",
    csv,
    "ranked_candidates.csv",
    "text/csv"
)