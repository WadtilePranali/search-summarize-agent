---
title: Search Summarize Agent
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🔎 Task-Automation Agent (Experiment 10)

An MLOps-grade task-automation agent that accepts an input topic, executes an automated web search across DuckDuckGo, and synthesizes the findings into an executive research brief of **exactly three bullet points** using Groq LLM inference.

## 🚀 Key Architectural Features
- **Dynamic Groq Model Selection:** Automatically resolves available models (`llama-3.3-70b-versatile`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `qwen/qwen3-32b`, `llama-3.1-8b-instant`) with automatic fallback.
- **Strict Formatting & Retries:** Normalizes bullet characters and verifies exact 3-bullet output with an automatic single retry if required.
- **Enterprise MLOps Pipeline:** GitHub Actions CI/CD for automated testing, container build validation, and zero-downtime deployment to Hugging Face Spaces.
- **Zero Secrets in Code:** Runtime secret injection with full audit proofing.

## ⚙️ Local Development
```bash
# 1. Clone repository
git clone <repo-url>
cd search-summarize-agent

# 2. Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variable
export GROQ_API_KEY="your_api_key_here"  # On Windows: $env:GROQ_API_KEY="..."

# 5. Launch application
streamlit run app.py
```

## 🐳 Docker Deployment
```bash
# Build Docker image
docker build -t search-summarize-agent .

# Run container with runtime secret injection
docker run -p 7860:7860 -e GROQ_API_KEY="your_api_key_here" search-summarize-agent
```
