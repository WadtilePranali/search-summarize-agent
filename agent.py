import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from ddgs import DDGS
import groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("agent")

# Model fallback priority candidates
DEFAULT_MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
]


def get_groq_model(api_key: Optional[str] = None) -> str:
    """
    Selects the Groq model:
    1. Checks the GROQ_MODEL environment variable.
    2. If not set, queries Groq models API with the provided key and picks
       the first available model matching DEFAULT_MODEL_CANDIDATES.
    3. Falls back to llama-3.1-8b-instant if dynamic lookup fails.
    """
    env_model = os.getenv("GROQ_MODEL", "").strip()
    if env_model:
        logger.info(f"Model selected from GROQ_MODEL env var: {env_model}")
        return env_model

    key = api_key or os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        logger.warning("GROQ_API_KEY not set during model resolution; defaulting to llama-3.1-8b-instant")
        return "llama-3.1-8b-instant"

    try:
        client = groq.Groq(api_key=key)
        models_resp = client.models.list()
        available_ids = [m.id for m in models_resp.data]
        logger.info(f"Found {len(available_ids)} models in Groq API")

        for candidate in DEFAULT_MODEL_CANDIDATES:
            if candidate in available_ids:
                logger.info(f"Selected priority fallback model: {candidate}")
                return candidate

        fallback = available_ids[0] if available_ids else "llama-3.1-8b-instant"
        logger.info(f"No priority candidate matched; defaulting to: {fallback}")
        return fallback
    except Exception as e:
        logger.warning(f"Error querying Groq models API: {e}. Defaulting to llama-3.1-8b-instant", exc_info=True)
        return "llama-3.1-8b-instant"


def search_web(topic: str, max_results: int = 6) -> str:
    """
    Stage 1: Uses DDGS().text() to perform web search.
    Returns numbered blocks (title, snippet, source URL) as text.
    On failure or no results, returns a string starting with 'SEARCH_ERROR'.
    """
    t0 = time.perf_counter()
    logger.info(f"Stage 1 starting: Searching web for topic '{topic}' (max_results={max_results})")
    try:
        ddgs = DDGS()
        raw_results = list(ddgs.text(topic, max_results=max_results))
        stage1_duration = time.perf_counter() - t0

        if not raw_results:
            logger.warning(f"Stage 1: No results found for topic '{topic}' ({stage1_duration:.3f}s)")
            return "SEARCH_ERROR: No search results found for the given topic."

        blocks = []
        for idx, item in enumerate(raw_results, start=1):
            title = item.get("title", "No Title")
            snippet = item.get("body", "No Snippet")
            url = item.get("href", "No URL")
            blocks.append(f"{idx}. Title: {title}\n   Snippet: {snippet}\n   Source: {url}")

        result_text = "\n\n".join(blocks)
        logger.info(f"Stage 1 completed in {stage1_duration:.3f}s. Retrieved {len(raw_results)} results.")
        return result_text
    except Exception as e:
        stage1_duration = time.perf_counter() - t0
        logger.error(f"Stage 1 failed after {stage1_duration:.3f}s: {e}", exc_info=True)
        return f"SEARCH_ERROR: Failed to retrieve search results - {e}"


def summarize(
    topic: str,
    results: str,
    temperature: float = 0.2,
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Stage 2: LangChain chain (ChatPromptTemplate | ChatGroq | StrOutputParser).
    Generates exactly three bullet points from the search results.
    """
    t0 = time.perf_counter()
    key = api_key or os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        logger.error("Stage 2: GROQ_API_KEY is not set.")
        raise ValueError("GROQ_API_KEY environment variable is missing.")

    selected_model = model or get_groq_model(api_key=key)
    logger.info(f"Stage 2 starting: Summarizing topic '{topic}' using model '{selected_model}' with temperature={temperature}")

    try:
        system_prompt = (
            "Using ONLY the search results, write EXACTLY three bullet points "
            "starting with '- ', one sentence each (max 30 words), no intro or conclusion, "
            "do not invent facts."
        )
        human_prompt = "Topic: {topic}\n\nSearch Results:\n{results}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt),
        ])

        llm = ChatGroq(
            model=selected_model,
            temperature=temperature,
            api_key=key,
        )

        chain = prompt | llm | StrOutputParser()
        output = chain.invoke({"topic": topic, "results": results})
        stage2_duration = time.perf_counter() - t0
        logger.info(f"Stage 2 completed in {stage2_duration:.3f}s using model '{selected_model}'")
        return output
    except Exception as e:
        stage2_duration = time.perf_counter() - t0
        logger.error(f"Stage 2 failed after {stage2_duration:.3f}s: {e}", exc_info=True)
        raise


def clean_to_three_bullets(text: str) -> List[str]:
    """
    Filters text: keeps only lines starting with -, •, or *,
    normalizes the marker to '- ', and caps at 3 bullets.
    """
    bullets: List[str] = []
    if not text:
        return bullets

    for line in text.strip().splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        # Check starting marker: -, *, • (\u2022)
        if line_clean.startswith(("-", "*", "•", "\u2022")):
            content = line_clean.lstrip("-*•\u2022").strip()
            if content:
                bullets.append(f"- {content}")
                if len(bullets) == 3:
                    break
    return bullets


def research_brief(
    topic: str,
    on_progress: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Executes the two-stage research pipeline:
    1. Search web with progress updates
    2. Summarize findings into exactly three bullet points
    3. Retries LLM once if fewer or more than 3 bullets are extracted
    """
    total_t0 = time.perf_counter()
    logger.info(f"Research brief initiated for topic: '{topic}'")

    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        logger.error("GROQ_API_KEY environment variable is not configured.")
        raise ValueError("GROQ_API_KEY is not set. Please set the GROQ_API_KEY environment variable.")

    # Stage 1: Search
    if on_progress:
        on_progress("Searching the web...")
    search_t0 = time.perf_counter()
    search_results = search_web(topic)
    search_duration = time.perf_counter() - search_t0

    if search_results.startswith("SEARCH_ERROR"):
        logger.error(f"Pipeline stopped due to search error: {search_results}")
        raise RuntimeError(search_results)

    # Stage 2: Summarize
    if on_progress:
        on_progress("Summarizing...")
    model_name = get_groq_model(api_key=key)
    summary_t0 = time.perf_counter()
    raw_summary = summarize(topic, search_results, temperature=0.2, model=model_name, api_key=key)
    summary_duration = time.perf_counter() - summary_t0

    bullets = clean_to_three_bullets(raw_summary)

    # Retry LLM once if not exactly 3 bullets
    if len(bullets) != 3:
        logger.warning(f"Initial summary produced {len(bullets)} bullets instead of 3. Retrying LLM once...")
        if on_progress:
            on_progress("Refining summary (retry)...")
        retry_t0 = time.perf_counter()
        raw_summary = summarize(topic, search_results, temperature=0.3, model=model_name, api_key=key)
        summary_duration += time.perf_counter() - retry_t0
        bullets = clean_to_three_bullets(raw_summary)
        logger.info(f"Retry finished. Final bullet count: {len(bullets)}")

    total_duration = time.perf_counter() - total_t0
    logger.info(
        f"Research brief completed in {total_duration:.3f}s for topic '{topic}'. "
        f"Model: {model_name}, Stage1: {search_duration:.3f}s, Stage2: {summary_duration:.3f}s, Bullets: {len(bullets)}"
    )

    return {
        "topic": topic,
        "bullets": bullets,
        "model": model_name,
        "stage1_time": search_duration,
        "stage2_time": summary_duration,
        "total_time": total_duration,
        "search_results": search_results,
        "raw_summary": raw_summary,
    }
