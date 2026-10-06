import os

from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_core.tools import tool
except Exception:  # pragma: no cover - optional dependency fallback
    def tool(*args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator

try:
    import requests
except Exception:  # pragma: no cover - optional dependency fallback
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency fallback
    BeautifulSoup = None

try:
    from tavily import TavilyClient
except Exception:  # pragma: no cover - optional dependency fallback
    TavilyClient = None

try:
    from rich import print
except Exception:  # pragma: no cover - optional dependency fallback
    def print(*args, **kwargs):
        return None


tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if TavilyClient is not None else None


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns titles, urls and snippets."""
    if tavily is None:
        return (
            "Dummy search result. Live Tavily access is unavailable.\n\n"
            f"Query: {query}\n"
            "1. Example source: https://example.com/source-1\n"
            "2. Example source: https://example.com/source-2"
        )

    results = tavily.search(query=query, max_results=5)
    out = []

    for r in results.get('results', []):
        out.append(
            f"Title: {r.get('title', 'Untitled')}\nURL: {r.get('url', 'unknown')}\nSnippet: {str(r.get('content', ''))[:300]}\r"
        )

    return "\n------\n".join(out) if out else "No search results returned."


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    if requests is None or BeautifulSoup is None:
        return (
            "Dummy scrape result. Remote page fetch is unavailable in this environment.\n"
            f"URL: {url}\n"
            "This simulated excerpt represents the content that would be extracted from a trusted source."
        )

    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"

    