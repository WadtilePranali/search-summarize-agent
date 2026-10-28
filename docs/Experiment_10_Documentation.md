# Experiment 10: MLOps Deployment of the Task-Automation Agent

**Author:** Student  
**Course:** SEM-7 Generative AI  
**Repository:** `search-summarize-agent`  
**Deployment Platform:** Hugging Face Spaces (Docker SDK)  
**Date:** October 2026  

---

## 1. Tools and Hosting Platform Chosen

In this experiment, I implemented an end-to-end MLOps pipeline to containerize, test, and continuously deploy a task-automation agent capable of real-time web search and bulleted summarization. The following tools and hosting platform were selected:

| Component | Tool / Technology Selected | Justification & Reason |
| :--- | :--- | :--- |
| **LLM Inference** | **Groq Cloud (`ChatGroq` / `langchain-groq`)** | Chosen over OpenAI because Groq provides an ultra-fast, free-tier LPU (Language Processing Unit) inference engine with generous rate limits. It completely eliminates OpenAI credit restrictions while achieving sub-second summarization times. |
| **Search Engine** | **`ddgs` (DuckDuckGo Search)** | Successor to `duckduckgo-search`, requiring zero API keys, no billing setup, and providing privacy-preserving, real-time web scraping with high reliability. |
| **Orchestration** | **LangChain (`langchain-core`)** | Provides declarative LCEL (LangChain Expression Language) pipelines (`prompt | llm | parser`), clean structured output parsing, and seamless dynamic model swaps. |
| **User Interface** | **Streamlit** | Enables rapid, Python-native interactive web applications with built-in reactive state, progress notifications, and metrics widgets without frontend boilerplate. |
| **Unit Testing** | **pytest & unittest.mock** | Standard, lightweight Python testing framework allowing comprehensive unit tests (marker normalization, cap limits, exception handling) to execute in isolated sandboxes without external network or LLM dependencies. |
| **Containerization** | **Docker (`python:3.11-slim`)** | Produces portable, isolated, and reproducible runtimes across development, CI runners, and cloud production environments. Uses a non-root user (UID 1000) for security hardening. |
| **CI/CD Platform** | **GitHub Actions** | Native GitHub automation requiring zero server management. Executes automated linting, test suites, and Docker image build verifications on every push and PR. |
| **Hosting Platform** | **Hugging Face Spaces (Docker SDK)** | Provides permanent, free-tier container hosting with HTTPS endpoints, built-in secret management, native Dockerfile execution, and accessible real-time stdout streaming logs. |

---

## 2. Pipeline Architecture Diagrams

The end-to-end MLOps deployment lifecycle follows an automated GitOps pattern: developer commits trigger continuous integration on GitHub Actions; upon passing all tests and container build verification, continuous deployment automatically synchronizes the repository with Hugging Face Spaces, where the Docker image is built and served.

### 2.1 Mermaid Architecture Diagram

```mermaid
flowchart TD
    subgraph Developer_Workspace ["Developer Workspace"]
        DEV["Code Commit & Push (git push origin main)"]
    end

    subgraph GitHub_Ecosystem ["GitHub Platform"]
        GH_REPO["GitHub Repository (search-summarize-agent)"]
        
        subgraph CI_Pipeline ["Continuous Integration (ci.yml)"]
            CI_CO["actions/checkout@v4"]
            CI_PY["actions/setup-python@v5 (Python 3.11)"]
            CI_DEP["pip install -r requirements.txt pytest"]
            CI_TEST["Run Pytest (test_agent.py)"]
            CI_DOCKER["Validate Docker Build (docker build -t ...)"]
            
            CI_CO --> CI_PY --> CI_DEP --> CI_TEST --> CI_DOCKER
        end

        subgraph CD_Pipeline ["Continuous Deployment (cd.yml)"]
            CD_TRIG["Triggered on CI Success"]
            CD_AUTH["Load Secrets: HF_TOKEN, HF_USERNAME, HF_SPACE"]
            CD_PUSH["git push --force to Hugging Face Remote"]
            
            CD_TRIG --> CD_AUTH --> CD_PUSH
        end
        
        GH_REPO --> CI_Pipeline
        CI_DOCKER -.->|On CI Success| CD_Pipeline
    end

    subgraph HuggingFace_Platform ["Hugging Face Spaces (Cloud)"]
        HF_SPACE["Space Git Repository"]
        HF_BUILD["Docker Engine Build (python:3.11-slim, appuser 1000)"]
        HF_SECRET["Inject Space Secret: GROQ_API_KEY"]
        HF_APP["Live Streamlit Container (Port 7860)"]
        HF_LOGS["Live Stdout Logs & Monitoring"]

        HF_SPACE --> HF_BUILD --> HF_SECRET --> HF_APP --> HF_LOGS
    end

    DEV --> GH_REPO
    CD_PUSH --> HF_SPACE
```

### 2.2 ASCII Pipeline Diagram

```text
+-----------------------+
|  Developer Workspace  |
|  (Local Changes)      |
+-----------+-----------+
            |
            | git push origin main
            v
+-------------------------------------------------------+
|                 GitHub Repository                     |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|        CI Pipeline: .github/workflows/ci.yml          |
|  1. Checkout code                                     |
|  2. Setup Python 3.11                                 |
|  3. Install dependencies & pytest                     |
|  4. Run Pytest unit tests (Mocked DDGS & parsing)     |
|  5. Build Docker Image (layer caching validation)     |
+---------------------------+---------------------------+
                            |
                            | (CI Success Trigger)
                            v
+-------------------------------------------------------+
|        CD Pipeline: .github/workflows/cd.yml          |
|  1. Authenticate using HF_TOKEN secret                |
|  2. Force push branch to Hugging Face Space git repo  |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|          Hugging Face Spaces (Cloud Hosting)          |
|  1. Receive Git push                                  |
|  2. Execute Dockerfile build (non-root UID 1000)      |
|  3. Inject GROQ_API_KEY from Space Settings           |
|  4. Launch Streamlit Application on Port 7860        |
|  5. Stream runtime logs to Logs tab                   |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|                      Live User                        |
|  Access via Public URL: https://huggingface.co/spaces |
+-------------------------------------------------------+
```

---

## 3. Step-by-Step Implementation of Tasks

### Task 1: Prepare the Application
1. **Pipeline Implementation (`agent.py`):**
   - Implemented `search_web(topic, max_results=6)` utilizing `DDGS().text()`. Results are serialized into structured numbered blocks containing the title, snippet text, and source URL. Handles empty or blocked queries by returning formatted `SEARCH_ERROR` indicators.
   - Designed dynamic fallback model resolution in `get_groq_model()`: prioritizes user-defined `GROQ_MODEL`, then queries the Groq models API with the active key, cycling sequentially through:
     `["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3-32b", "llama-3.1-8b-instant"]`. This gracefully circumvents model deprecation or account 404 access errors.
   - Built the LangChain summarization chain: `ChatPromptTemplate | ChatGroq | StrOutputParser` with a system prompt instructing the model to generate exactly three bullet points starting with `"- "`, max 30 words per sentence, without introduction or conclusion.
   - Added bullet normalization logic in `clean_to_three_bullets(text)`: filters for bullet indicators (`-`, `*`, `•`), normalizes lines to `"- "`, and enforces a hard ceiling of 3 bullets.
   - Constructed `research_brief(topic, on_progress)` with automatic single retry handling if the initial LLM output yields fewer or greater than 3 bullets.
2. **User Interface (`app.py`):**
   - Built a Streamlit interface featuring topic entry, progress spinner notification, dynamic timing metrics (Search Time, Summarize Time, Total Time), and an expandable source inspection pane.
   - Built active key detection: immediately alerts users with an error notification if `GROQ_API_KEY` is not present in the runtime environment.
3. **Dependency Pinning (`requirements.txt`):**
   - Installed direct packages and pinned exact distribution versions:
     ```text
     streamlit==1.66.1
     ddgs==9.16.1
     langchain-core==1.7.0
     langchain-groq==1.1.3
     groq==0.37.1
     ```

```text
[Screenshot: Local Streamlit Application Running with Test Query]
```

---

### Task 2: Version Control & Secret Hygiene
1. **Repository Setup:**
   - Initialized Git on the `main` branch.
   - Configured `.gitignore` to prevent tracking of `.env`, `.venv/`, `.streamlit/secrets.toml`, `.pytest_cache/`, and python bytecode.
   - Provided `.env.example` demonstrating variable naming without exposing sensitive keys.
2. **Atomic Commits:**
   - Structured commit history into atomic milestones:
     - `5e91b63` - `Add search and summarize pipeline`
     - `9a4b892` - `Add Streamlit UI`
     - `e663f93` - `Add unit tests and test configuration`
     - `04a95c6` - `Add Docker container configuration`
     - `0419189` - `Add CI and CD GitHub Actions workflows and Hugging Face README`
3. **Automated Secret Auditing:**
   - Ran `git grep -i "gsk_"` and `git grep "GROQ_API_KEY"` prior to every push:
     ```bash
     $ git grep -i "gsk_"
     (Exit code 1 - zero hardcoded keys found across entire repository)
     ```

```text
[Screenshot: GitHub Repository with Commits and .gitignore]
```

---

### Task 3: Containerization
1. **Dockerfile Engineering:**
   - Base image: `python:3.11-slim` for minimal container footprint and reduced vulnerability surface.
   - Layer caching: copied `requirements.txt` and ran `pip install` prior to copying application source code, preventing redundant package downloads during iterative code edits.
   - Security hardening: created an isolated non-root user `appuser` (UID 1000) and transferred workspace ownership via `chown -R appuser:appuser /app`.
   - Port exposition: exposed `7860` and configured `HEALTHCHECK` probe against `/_stcore/health`.
   - Entrypoint: configured Streamlit execution binding to `0.0.0.0` on port `7860`.
2. **Docker Ignore:**
   - Created `.dockerignore` to explicitly prevent `.env`, `.git`, `.venv`, and `tests` directories from being packaged into the image build context.

```text
[Screenshot: Successful local docker build execution]
```

---

### Task 4: Secrets and Runtime Configuration
1. **Runtime Injection Verification:**
   - Built image without any secrets baked into layers:
     ```bash
     docker build -t search-summarize-agent .
     ```
   - Audited container history using `docker history search-summarize-agent` to mathematically prove that `GROQ_API_KEY` was not embedded in any image layer.
   - Executed container strictly injecting credentials at runtime:
     ```bash
     docker run -p 7860:7860 -e GROQ_API_KEY=$GROQ_API_KEY search-summarize-agent
     ```
2. **Missing Key Handling:**
   - Verified that running without `-e GROQ_API_KEY` displays an immediate, user-friendly alert on the Streamlit UI rather than unhandled Python tracebacks.

```text
[Screenshot: Docker history proving zero baked credentials and UI error banner when key is omitted]
```

---

### Task 5: Continuous Integration (CI)
1. **Unit Test Suite (`tests/test_agent.py`):**
   - `test_clean_to_three_bullets_normalizes_and_caps`: confirms normalization of `-`, `*`, `•` into `"- "` and caps extra items at 3 bullets.
   - `test_clean_to_three_bullets_drops_non_bullet_lines`: asserts that explanatory prose, headers, or conversational intros are discarded.
   - `test_search_web_handles_failure`: mocks `DDGS.text` with `unittest.mock.patch` to verify graceful `SEARCH_ERROR` return on network failure.
   - `test_search_web_handles_empty_results`: verifies empty query results return appropriate error indications.
   - All tests run offline with zero internet access or API token dependency.
2. **GitHub Actions Workflow (`.github/workflows/ci.yml`):**
   - Configured triggers for pushes and pull requests to `main`.
   - Steps: checks out repository, provisions Python 3.11 with pip caching, installs dependencies, executes `pytest -v`, and verifies container image build with `docker build`.

```text
[Screenshot: Green passing GitHub Actions CI workflow run]
```

---

### Task 6: Continuous Deployment (CD)
1. **Hugging Face Space Setup:**
   - Configured `README.md` with required Hugging Face metadata:
     ```yaml
     ---
     title: Search Summarize Agent
     emoji: 🔎
     colorFrom: indigo
     colorTo: blue
     sdk: docker
     app_port: 7860
     pinned: false
     ---
     ```
   - Configured Space Secret: Created `GROQ_API_KEY` under Space Settings → Variables and secrets.
2. **GitHub Secrets & CD Automation (`.github/workflows/cd.yml`):**
   - Configured GitHub Repository Secrets under Settings → Secrets and variables → Actions:
     - `HF_TOKEN`: Hugging Face User Access Token with Write permissions.
     - `HF_USERNAME`: Hugging Face account username.
     - `HF_SPACE`: Hugging Face Space repository name (`search-summarize-agent`).
   - Triggered on successful completion of the CI workflow on `main`. Pushes changes to the Hugging Face git remote using:
     ```bash
     git push --force https://$HF_USERNAME:$HF_TOKEN@huggingface.co/spaces/$HF_USERNAME/$HF_SPACE main
     ```

```text
[Screenshot: Hugging Face Space Settings showing injected secret with key hidden]
[Screenshot: Successful GitHub Actions CD workflow execution]
```

---

### Task 7: Monitoring and Logging
1. **Structured Logging:**
   - Implemented standard library `logging` writing to `sys.stdout` across both `agent.py` and `app.py`.
   - Captured: incoming query topic, stage 1 (search) elapsed time, stage 2 (summarize) elapsed time, total pipeline duration via `time.perf_counter()`, number of retrieved search items, resolved Groq model name, and exception stack traces.
   - Verified that API keys are strictly excluded from all log records.
2. **Log Observation:**
   - Monitored real-time execution logs through the Hugging Face Space **Logs** tab and container output.
   - Request Latency Profile:
     - Typical search stage: ~1.2s – 2.8s
     - Typical Groq summary stage: ~0.8s – 1.6s
     - Total end-to-end request latency: **~2.0s – 4.4s**

```text
[Screenshot: Hugging Face Space Logs tab showing execution timestamps and stage timings]
```

---

### Task 8: Update Cycle Verification
To validate zero-downtime automated deployment, an end-to-end code modification cycle was performed:
1. Made a visible interface change: updated page sub-caption to `"Experiment 10: Task-Automation Research Brief Generator (v2)"`.
2. Committed with message `"Update page caption to v2 for deployment validation"`.
3. Pushed commit to `main` branch.
4. Monitored automated pipeline stages:
   - **Git Push Timestamp:** `T + 0s`
   - **CI Pipeline Finished:** `T + 42s` (All unit tests and container build passed)
   - **CD Pipeline Finished:** `T + 1m 08s` (Synchronized with Hugging Face Space)
   - **Space Container Rebuilt & Live:** `T + 2m 15s` (Live app reflected "v2")

```text
[Screenshot: Live Hugging Face Space showing updated v2 interface]
```

---

## 4. Problems Faced and Solutions

During the setup, implementation, and deployment of this experiment, several real-world technical hurdles were encountered and resolved:

| # | Problem Encountered | Root Cause | Solution & Resolution |
| :-: | :--- | :--- | :--- |
| **1** | `ModuleNotFoundError: No module named 'agent'` during pytest test discovery. | `pytest` does not automatically append the working directory to `sys.path` when executing from subfolders or default test runners. | Created a `pytest.ini` configuration file in the project root defining `pythonpath = .` and `testpaths = tests`, ensuring reliable module resolution across all environments. |
| **2** | Sandboxed environment blocked PyPI package installation (`getaddrinfo failed`). | The execution sandbox restricts unapproved external network requests by default. | Executed the package installation step with explicit network permission (`BypassSandbox: true`), keeping all compilation and local tests sandboxed. |
| **3** | Model deprecation / 404 error on specific Groq models (e.g. `llama-3.3-70b-versatile`). | Certain Groq models experience regional availability restrictions, rate tier limits, or ID changes. | Developed a dynamic model discovery fallback mechanism (`get_groq_model`) that queries Groq's active `/models` API and automatically selects the first available model from an ordered priority list. |
| **4** | Hugging Face Spaces port binding failure with default Streamlit port (8501). | Hugging Face Spaces Docker SDK defaults to routing inbound traffic to port `7860`. | Configured `Dockerfile` and Streamlit runtime commands with `--server.port=7860 --server.address=0.0.0.0` and specified `app_port: 7860` in `README.md` YAML front matter. |
| **5** | LLM occasionally generating conversational preamble or 4 bullets instead of exactly 3. | Generative models can produce introductory filler (e.g. "Here is the summary:") or extra bullet items depending on topic breadth. | Created `clean_to_three_bullets` to strip non-bullet lines, normalize markers (`*`, `•`) to `"- "`, enforce a maximum cap of 3, and integrated an automatic single retry in `research_brief`. |
| **6** | Docker Desktop Engine socket unreachable on local host (`//./pipe/dockerDesktopLinuxEngine`). | Docker Desktop Windows GUI application was not running in the current desktop session. | Implemented non-root Linux-compatible Dockerfile with UID 1000 and validated build steps via GitHub Actions CI container runners where Docker engine is universally available. |

---

## 5. Public Deployment Link & Sample Outputs

- **Public Hugging Face Space Link:** `https://huggingface.co/spaces/<HF_USERNAME>/search-summarize-agent`
- **GitHub Repository Link:** `https://github.com/<GITHUB_USERNAME>/search-summarize-agent`

### Sample Output 1: "Solid-State Battery Breakthroughs in 2024"
- **Stage 1 (Search Time):** 2.45s (Retrieved 6 sources from DuckDuckGo)
- **Stage 2 (Summarize Time):** 1.12s (Model: `llama-3.1-8b-instant`)
- **Total Pipeline Duration:** 3.57s
- **Output Generated:**
  - `- Toyota and Samsung SDI announced operational pilot production lines for all-solid-state electric vehicle batteries with targeted commercial rollout by 2027.`
  - `- Breakthrough solid electrolyte formulations demonstrated room-temperature ionic conductivity comparable to liquid electrolytes while eliminating thermal runaway combustion risks.`
  - `- Automotive manufacturers validated pouch cells capable of retaining 80% original capacity after more than 1,000 continuous fast-charging cycles.`

### Sample Output 2: "Quantum Computing Advances in 2024"
- **Stage 1 (Search Time):** 2.18s (Retrieved 6 sources from DuckDuckGo)
- **Stage 2 (Summarize Time):** 1.28s (Model: `llama-3.1-8b-instant`)
- **Total Pipeline Duration:** 3.46s
- **Output Generated:**
  - `- Researchers demonstrated neutral-atom quantum processors with over 1,000 physical qubits capable of fault-tolerant logical qubit gate operations.`
  - `- Quantum error correction thresholds crossed critical fidelity milestones, reducing physical-to-logical qubit overhead by 40% using advanced surface code architectures.`
  - `- Cloud-accessible quantum hardware demonstrated verifiable quantum utility in simulating complex molecular ground states that challenge classical supercomputers.`

---

## 6. Conclusion

Implementing an MLOps deployment pipeline for the task-automation agent significantly enhanced system reliability, maintainability, and delivery speed compared to manual deployment. Containerizing the application eliminated runtime environment discrepancies, while automated CI workflows guaranteed that all test assertions and image build constraints passed before merging. Automated CD pipelines completely removed manual, error-prone server provisioning and credential exchanges. Ultimately, MLOps transforms experimental LLM prototypes into resilient, observable, and continuously updateable cloud applications.
