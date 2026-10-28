import os
import sys
import time
import logging
import streamlit as st

# Attempt to load local .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agent import research_brief

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("app")

# Page Configuration
st.set_page_config(
    page_title="Task-Automation Agent",
    page_icon="🔎",
    layout="centered"
)

# Custom header styling
st.title("🔎 Search & Summarize Agent")
st.caption("Experiment 10: Task-Automation Research Brief Generator")

# Check for GROQ_API_KEY
groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

if not groq_api_key:
    st.error(
        "⚠️ **Configuration Error: `GROQ_API_KEY` is missing!**\n\n"
        "Please provide your Groq API key via environment variable `GROQ_API_KEY` "
        "(e.g., in Hugging Face Space Settings → Secrets, Docker `-e GROQ_API_KEY=...`, or local `.env`)."
    )

st.markdown(
    "Enter any topic to conduct an automated real-time web search and receive "
    "an AI-generated executive brief of **exactly three bullet points**."
)

topic = st.text_input(
    "Topic to research:",
    placeholder="e.g. Advancements in Solid-State Batteries in 2024",
    disabled=not groq_api_key
)

generate_btn = st.button("Generate Brief", type="primary", disabled=not groq_api_key)

if generate_btn:
    if not topic.strip():
        st.warning("Please enter a non-empty topic to generate a brief.")
    else:
        req_start = time.perf_counter()
        logger.info(f"UI request received for topic: '{topic.strip()}'")

        # Status / progress indicator container
        status_placeholder = st.empty()
        
        def update_progress(message: str):
            status_placeholder.info(f"⏳ {message}")

        try:
            with st.spinner("Executing task-automation pipeline..."):
                result = research_brief(topic.strip(), on_progress=update_progress)

            total_ui_time = time.perf_counter() - req_start
            status_placeholder.empty()

            st.success("✅ Research brief generated successfully!")

            st.subheader("📋 Executive Summary (3 Key Highlights)")
            for bullet in result["bullets"]:
                st.markdown(f"**{bullet}**")

            # Metrics display
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Search Time", value=f"{result['stage1_time']:.2f}s")
            with col2:
                st.metric(label="Summarize Time", value=f"{result['stage2_time']:.2f}s")
            with col3:
                st.metric(label="Total Pipeline Time", value=f"{result['total_time']:.2f}s")

            # Details expander
            with st.expander("🔍 View Pipeline & Source Details"):
                st.write(f"**Model Used:** `{result['model']}`")
                st.markdown("#### Raw Search Results")
                st.code(result["search_results"], language="markdown")

            logger.info(
                f"UI request succeeded for '{topic.strip()}'. Total UI time: {total_ui_time:.3f}s. "
                f"Model: {result['model']}. Bullets returned: {len(result['bullets'])}"
            )

        except Exception as e:
            total_ui_time = time.perf_counter() - req_start
            status_placeholder.empty()
            logger.error(
                f"UI request failed for topic '{topic.strip()}' after {total_ui_time:.3f}s: {e}",
                exc_info=True
            )
            st.error(f"❌ **An error occurred during pipeline execution:**\n\n{e}")
