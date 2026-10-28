import pytest
from unittest.mock import patch, MagicMock
from agent import clean_to_three_bullets, search_web


def test_clean_to_three_bullets_normalizes_and_caps():
    """Test (a): clean_to_three_bullets returns max 3 bullets and normalizes markers to '- '."""
    raw_input = """
    * First item with asterisk
    • Second item with unicode bullet
    - Third item with dash
    * Fourth item should be excluded because of cap
    • Fifth item should be excluded because of cap
    """
    bullets = clean_to_three_bullets(raw_input)
    assert len(bullets) == 3, f"Expected 3 bullets, got {len(bullets)}"
    for b in bullets:
        assert b.startswith("- "), f"Expected bullet to start with '- ', got '{b}'"
    assert bullets[0] == "- First item with asterisk"
    assert bullets[1] == "- Second item with unicode bullet"
    assert bullets[2] == "- Third item with dash"


def test_clean_to_three_bullets_drops_non_bullet_lines():
    """Test (b): clean_to_three_bullets drops lines that do not start with a bullet marker."""
    raw_input = """
    Here is an executive brief based on the web search results:
    In recent years, many breakthroughs have occurred.

    - Renewable energy adoption increased globally by 15%.
    Notice this explanatory paragraph right in the middle.
    - Solar and wind energy represent the majority of new capacity additions.
    And a concluding sentence here.
    - Battery energy storage capacity doubled across major grids.
    Another trailing thought.
    """
    bullets = clean_to_three_bullets(raw_input)
    assert len(bullets) == 3
    assert bullets[0] == "- Renewable energy adoption increased globally by 15%."
    assert bullets[1] == "- Solar and wind energy represent the majority of new capacity additions."
    assert bullets[2] == "- Battery energy storage capacity doubled across major grids."


@patch("agent.DDGS")
def test_search_web_handles_failure(mock_ddgs_class):
    """Test (c): search_web returns a string starting with SEARCH_ERROR when search fails."""
    mock_instance = MagicMock()
    mock_instance.text.side_effect = Exception("DuckDuckGo rate limit or connection timeout")
    mock_ddgs_class.return_value = mock_instance

    result = search_web("quantum computing")
    assert isinstance(result, str)
    assert result.startswith("SEARCH_ERROR"), f"Expected result to start with SEARCH_ERROR, got: {result}"
    assert "DuckDuckGo rate limit or connection timeout" in result


@patch("agent.DDGS")
def test_search_web_handles_empty_results(mock_ddgs_class):
    """Test that search_web handles empty results by returning SEARCH_ERROR."""
    mock_instance = MagicMock()
    mock_instance.text.return_value = []
    mock_ddgs_class.return_value = mock_instance

    result = search_web("nonexistent topic xyz12345")
    assert result.startswith("SEARCH_ERROR")
    assert "No search results found" in result
