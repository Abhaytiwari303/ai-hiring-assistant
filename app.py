import streamlit as st
import requests
import re
import pdfplumber
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords once
nltk.download('stopwords')

# Streamlit Page Config
st.set_page_config(page_title="AI Hiring Assistant", layout="centered")
st.title("🤖 TalentScout - AI Hiring Assistant (Powered by LLaMA)")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "candidate_info" not in st.session_state:
    st.session_state.candidate_info = {}
if "tech_questions" not in st.session_state:
    st.session_state.tech_questions = []
if "tech_answers" not in st.session_state:
    st.session_state.tech_answers = []
if "chat_done" not in st.session_state:
    st.session_state.chat_done = False
if "asking_questions" not in st.session_state:
    st.session_state.asking_questions = False
if "show_resume" not in st.session_state:
    st.session_state.show_resume = False

# Function to ask local LLaMA model via Ollama API
def ask_llama(prompt):
    try:
        response = requests.post("http://127.0.0.1:11434/api/generate", json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        })
        res_json = response.json()
        if "response" in res_json:
            return res_json["response"]
        elif "choices" in res_json and len(res_json["choices"]) > 0:
            message = res_json["choices"][0].get("message", {})
            return message.get("content", "[LLaMA Error] No content found")
        else:
            return "[LLaMA Error] Unexpected response structure"
    except Exception as e:
        return f"[LLaMA Error] {str(e)}"

# Initial greeting if no messages yet
if not st.session_state.messages:
    greeting = "Hi there! 👋 I'm your Hiring Assistant. Let's get started.\nPlease provide your Full Name:"
    st.session_state.messages.append({"role": "assistant", "content": greeting})

# Display conversation messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Candidate info collection fields
expected_fields = ["Full Name", "Email", "Phone", "Years of Experience", "Desired Position", "Location", "Tech Stack"]

def get_next_field():
    for field in expected_fields:
        if field not in st.session_state.candidate_info or st.session_state.candidate_info[field].strip() == "":
            return field
    return None

def collect_candidate_info(user_input):
    user_input = user_input.strip()
    if user_input.lower() in ['exit', 'quit', 'stop']:
        st.session_state.chat_done = True
        return "Thank you! We'll contact you with the next steps. Goodbye!"

    if user_input == "":
        next_field = get_next_field()
        return f"Input cannot be empty. Please provide your {next_field}:"

    next_field = get_next_field()
    if next_field:
        st.session_state.candidate_info[next_field] = user_input
        next_field = get_next_field()
        if next_field:
            return f"Please provide your {next_field}:"
        else:
            return "Thanks! Generating technical questions based on your Tech Stack..."
    return None

# --- ATS Resume Parser Section ---
job_role_keywords = {
    "Data Scientist": [
        "python", "machine learning", "data science", "deep learning",
        "nlp", "tensorflow", "pytorch", "sql", "statistics",
        "data visualization", "pandas", "numpy", "scikit-learn", "r"
    ],
    "Software Engineer": [
        "java", "c++", "python", "git", "docker", "rest api",
        "microservices", "kubernetes", "aws", "sql", "agile", "spring",
        "javascript", "react", "node.js"
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "aws", "terraform", "jenkins", "ci/cd",
        "monitoring", "linux", "bash", "ansible", "prometheus", "grafana"
    ],
    "Frontend Developer": [
        "html", "css", "javascript", "react", "vue.js", "angular",
        "typescript", "webpack", "sass", "responsive design", "ui/ux"
    ],
    "Backend Developer": [
        "java", "python", "node.js", "express", "sql", "mongodb",
        "rest api", "microservices", "docker", "kubernetes", "aws"
    ],
    "Full Stack Developer": [
        "javascript", "react", "node.js", "express", "sql", "mongodb",
        "docker", "aws", "html", "css", "typescript", "graphql"
    ],
    "Machine Learning Engineer": [
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "scikit-learn", "nlp", "computer vision", "docker", "aws", "api"
    ],
    "Data Engineer": [
        "sql", "python", "spark", "hadoop", "airflow", "etl", "aws",
        "big data", "kafka", "data warehousing", "nosql"
    ],
    "QA Engineer": [
        "selenium", "test automation", "manual testing", "junit", "cucumber",
        "postman", "performance testing", "jira", "bug tracking", "rest api"
    ],
    "Cloud Engineer": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "ci/cd", "cloud architecture", "monitoring", "security"
    ],
    "Security Engineer": [
        "network security", "penetration testing", "firewalls", "encryption",
        "vulnerability assessment", "aws security", "compliance", "siem",
        "incident response"
    ],
    "Mobile Developer": [
        "android", "ios", "java", "kotlin", "swift", "react native",
        "flutter", "mobile ui/ux", "xcode", "android studio"
    ],
    "Database Administrator": [
        "sql", "oracle", "mysql", "postgresql", "performance tuning",
        "backup and recovery", "replication", "indexing", "nosql"
    ],
    "Product Manager": [
        "roadmap planning", "stakeholder management", "agile", "scrum",
        "user stories", "market research", "analytics", "communication"
    ],
    "UX/UI Designer": [
        "wireframing", "prototyping", "figma", "adobe xd", "user research",
        "interaction design", "visual design", "usability testing"
    ]
}


# Sidebar input for job role
job_role = st.sidebar.text_input("Enter Job Role (for ATS Keywords):", value="Data Scientist")
selected_keywords = job_role_keywords.get(job_role, job_role_keywords["Data Scientist"])

def extract_resume_text(uploaded_file):
    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join([page.extract_text() or '' for page in pdf.pages])
    return ""

def calculate_ats_score(resume_text, keywords):
    resume_text_lower = resume_text.lower()
    matched_keywords = [kw for kw in keywords if kw.lower() in resume_text_lower]
    score = (len(matched_keywords) / len(keywords)) * 100 if keywords else 0
    return score, matched_keywords

def parse_tech_questions(raw_response):
    questions = []
    lines = raw_response.split("\n")
    for line in lines:
        line = line.strip()
        line = re.sub(r"^(\d+[\.\)]|\-|\*)\s*", "", line)
        if line:
            questions.append(line)
    return questions

def clean_and_tokenize(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return [word for word in tokens if word not in stop_words]

def get_ats_score(resume_text, jd_text):
    if not resume_text or not jd_text:
        return 0, [], []
    resume_tokens = clean_and_tokenize(resume_text)
    jd_tokens = clean_and_tokenize(jd_text)
    resume_keywords = set(resume_tokens)
    jd_keywords = set(jd_tokens)
    matched = resume_keywords.intersection(jd_keywords)
    missing = jd_keywords.difference(resume_keywords)
    score = int(len(matched) / len(jd_keywords) * 100) if jd_keywords else 0
    return score, matched, missing

# Sidebar Resume Upload and Analysis
uploaded_resume = st.sidebar.file_uploader("Upload your Resume (PDF only)", type=["pdf"])
if uploaded_resume:
    resume_text = extract_resume_text(uploaded_resume)
    if resume_text:
        st.sidebar.success("✅ Resume successfully extracted!")
        ats_score, matched_keywords = calculate_ats_score(resume_text, selected_keywords)
        st.sidebar.subheader("📊 Resume ATS Summary")
        st.sidebar.markdown(f"**ATS Score:** {ats_score:.2f}%")
        st.sidebar.markdown(f"**Matched Keywords:** {', '.join(matched_keywords) if matched_keywords else 'None'}")

        def extract_name(text):
        # Consider only the first 10 non-empty lines
            lines = [line.strip() for line in text.split('\n') if line.strip()][:10]

            # Try to extract from a line containing "Name:"
            for line in lines:
                if 'name' in line.lower():
                    match = re.search(r"(?:name\s*[:\-]?\s*)([A-Z][a-z]+(?: [A-Z][a-z]+)+)", line, re.IGNORECASE)
                    if match:
                        return match.group(1)

            # Otherwise, use name-like heuristic (e.g., 2 words, capitalized, no numbers)
            for line in lines:
                if 2 <= len(line.split()) <= 4 and line.replace(" ", "").isalpha():
                    if all(word[0].isupper() for word in line.split()):
                        return line

            return None



        name = extract_name(resume_text)
        email_match = re.search(r"[\w\.-]+@[\w\.-]+", resume_text)
        phone_match = re.search(r"\+?\d[\d\s\-\(\)]{8,15}", resume_text)

        st.sidebar.markdown(f"**Name:** {name if name else 'Not found'}")
        st.sidebar.markdown(f"**Email:** {email_match.group(0) if email_match else 'Not found'}")
        st.sidebar.markdown(f"**Phone:** {phone_match.group(0) if phone_match else 'Not found'}")

        if st.sidebar.button("Show / Hide Full Resume Text"):
            st.session_state.show_resume = not st.session_state.show_resume
        if st.session_state.show_resume:
            st.sidebar.text_area("🧾 Full Resume Text (Parsed)", resume_text[:2000], height=300)

# ATS Compatibility Check with Job Description (optional)
if all(field in st.session_state.candidate_info for field in expected_fields):
    st.markdown("---")
    st.markdown("### 📄 ATS Compatibility Check (Optional)")
    resume_file = st.file_uploader("Upload your Resume (PDF only) for ATS Check", type=["pdf"])
    job_description = st.text_area("Paste the Job Description here (optional)", height=150)
    if resume_file and job_description:
        with st.spinner("Analyzing resume..."):
            resume_text_check = extract_resume_text(resume_file)
            score, matched_keywords, missing_keywords = get_ats_score(resume_text_check, job_description)
            st.success(f"✅ ATS Compatibility Score: **{score}%**")
            st.markdown(f"**✅ Matched Keywords ({len(matched_keywords)}):** {', '.join(list(matched_keywords)[:15])}")
            st.markdown(f"**⚠️ Missing Keywords ({len(missing_keywords)}):** {', '.join(list(missing_keywords)[:15])}")
            if score >= 80:
                st.success("👍 Your resume is highly ATS-friendly!")
            elif score >= 50:
                st.warning("🟡 Your resume is moderately ATS-compatible. Consider adding more relevant keywords.")
            else:
                st.error("❌ Your resume might get filtered out by ATS systems. Add more role-specific keywords.")

# Chat input and logic
prompt = st.chat_input("Type your response here...")

if prompt:
    # Restart command
    if prompt.lower() == "restart":
        if st.confirm("Are you sure you want to restart the chat? All data will be lost."):
            st.session_state.clear()
            st.experimental_rerun()
    else:
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not st.session_state.chat_done and not st.session_state.asking_questions:
            bot_response = collect_candidate_info(prompt)
            st.chat_message("assistant").write(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

            # After candidate info collected, generate questions
            if bot_response == "Thanks! Generating technical questions based on your Tech Stack...":
                tech_stack = st.session_state.candidate_info.get("Tech Stack", "")
                tech_prompt = (
                    f"You are an AI hiring assistant. Based on the candidate's tech stack ({tech_stack}), "
                    f"generate 3 challenging interview questions for each technology listed. "
                    f"Return questions as a numbered list."
                )
                with st.spinner("Generating technical questions..."):
                    raw_response = ask_llama(tech_prompt)
                questions = parse_tech_questions(raw_response)
                if not questions:
                    questions = ["Sorry, I couldn't generate questions at this time."]
                st.session_state.tech_questions = questions
                st.session_state.asking_questions = True

                st.chat_message("assistant").write("Here are your technical questions:")
                st.session_state.messages.append({"role": "assistant", "content": "Here are your technical questions:"})
                for q in questions:
                    st.chat_message("assistant").write(q)
                    st.session_state.messages.append({"role": "assistant", "content": q})

                st.chat_message("assistant").write("You can now answer these questions one by one, or type 'exit' to finish.")
                st.session_state.messages.append({"role": "assistant", "content": "You can now answer these questions one by one, or type 'exit' to finish."})

        elif st.session_state.asking_questions and not st.session_state.chat_done:
            if prompt.lower() in ['exit', 'quit', 'stop']:
                st.session_state.chat_done = True
                st.chat_message("assistant").write("Thank you for your time! We'll contact you with the next steps.")
                st.session_state.messages.append({"role": "assistant", "content": "Thank you for your time! We'll contact you with the next steps."})
            else:
                st.session_state.tech_answers.append(prompt)
                idx = len(st.session_state.tech_answers)
                if idx < len(st.session_state.tech_questions):
                    next_question = st.session_state.tech_questions[idx]
                    st.chat_message("assistant").write(f"Next question:\n{next_question}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Next question:\n{next_question}"})
                else:
                    st.session_state.chat_done = True
                    st.chat_message("assistant").write("Great! You have answered all the questions. Thank you!")
                    st.session_state.messages.append({"role": "assistant", "content": "Great! You have answered all the questions. Thank you!"})

                    # Show summary after chat done
                    st.markdown("---")
                    st.markdown("### 📄 Summary of Your Information")
                    for k, v in st.session_state.candidate_info.items():
                        st.markdown(f"**{k}**: {v}")
                    st.markdown("### 📝 Your Answers to Technical Questions")
                    for i, (q, a) in enumerate(zip(st.session_state.tech_questions, st.session_state.tech_answers), 1):
                        st.markdown(f"**Q{i}: {q}**")
                        st.markdown(f"**A{i}:** {a}")

        else:
            st.chat_message("assistant").write("Let me know if you want to restart or exit.")
            st.session_state.messages.append({"role": "assistant", "content": "Let me know if you want to restart or exit."})
