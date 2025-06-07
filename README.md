title: "AI Hiring Assistant"
emoji: "🧠💼"
description: "A local LLaMA 3-powered AI Hiring Assistant built with Streamlit to automate candidate screening and evaluation."

project_overview: |
  AI Hiring Assistant is a Streamlit web app designed to streamline the technical screening process.
  It:
    - Collects candidate information (name, skills, experience, etc.)
    - Uses LLaMA 3 via Ollama for generating role-specific technical questions
    - Optionally evaluates candidate answers
    - Summarizes candidate performance for recruiters to review
    -ATS Score using Resume with there role detail

installation_instructions:
  - step: "Clone the Repository"
    command: |
      git clone https://github.com/your-username/ai_hiring_assistant.git
      cd ai_hiring_assistant

  - step: "Create and Activate Virtual Environment"
    command: |
      python -m venv llama_env
      .\llama_env\Scripts\activate  # Windows

  - step: "Install Required Libraries"
    command: |
      pip install -r requirements.txt

  - step: "Install and Start Ollama"
    command: |
      Download and install Ollama from https://ollama.com
      ollama run llama3

  - step: "Run the Streamlit App"
    command: |
      streamlit run app.py

usage_guide: |
  1. Open the app in your browser.
  2. Fill in candidate details (name, experience, skills).
  3. Choose a job role (e.g., Python Developer).
  4. The assistant will:
     - Ask tailored technical questions via llama3 (local model)
     - Optionally assess candidate answers
     - Generate a recruiter-friendly report

technical_details:
  frontend: "Streamlit"
  llm: "LLaMA 3 (via Ollama - locally run)"
  backend_language: "Python"
  core_libraries:
    - streamlit
    - requests / subprocess (for Ollama API access)
    - python-dotenv
    - json
    - re
  architecture: |
    - Streamlit UI
    - Python backend handling user prompts and model calls
    - Local LLaMA 3 model via Ollama

prompt_design: |
  - Custom-designed prompts adjust dynamically to the user's job title, experience, and skills.
  - Prompt structure encourages deep, role-relevant technical questions.
  - Prompts simulate a human recruiter to maintain natural conversation flow.

challenges_and_solutions:
  - challenge: "Local model integration (LLaMA 3 via Ollama)"
    solution: "Used subprocess calls and prompt optimization to ensure smooth interaction"
  - challenge: "High initial latency from LLM"
    solution: "Cached static parts of prompts and minimized tokens"
  - challenge: "Prompt quality for technical depth"
    solution: "Iteratively refined prompts to ensure specificity and job relevance"

future_enhancements:
  - "Add resume upload and NLP-based parsing"
  - "Integrate scoring engine for auto-evaluation"
  - "Deploy via Docker or local GUI executable"
  - "Voice-based interaction with TTS and STT modules"

contributing: |
  Contributions are welcome! Fork the repo, make changes, and open a pull request.
  Please follow conventional commit messages and document your changes.

license: |
  This project is licensed under the MIT License.
  See the LICENSE file for full details.
