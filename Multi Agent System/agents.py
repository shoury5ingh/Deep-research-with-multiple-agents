import os

from dotenv import load_dotenv

load_dotenv()

try:
    from langchain.agents import create_agent
except Exception:  # pragma: no cover - optional dependency fallback
    create_agent = None

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
except Exception:  # pragma: no cover - optional dependency fallback
    StrOutputParser = None
    ChatPromptTemplate = None

try:
    from langchain_mistralai import ChatMistralAI
except Exception:  # pragma: no cover - optional dependency fallback
    ChatMistralAI = None

try:
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover - optional dependency fallback
    ChatOpenAI = None

from tools import web_search, scrape_url


def _build_llm():
    if ChatMistralAI is not None and os.getenv("MISTRAL_API_KEY"):
        return ChatMistralAI(
            model=os.getenv("MISTRAL_MODEL", "ministral-8b-2512"),
            temperature=0,
        )

    if ChatOpenAI is not None and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
        )

    return None


llm = _build_llm()


def _extract_text(message):
    if message is None:
        return ""
    if hasattr(message, "content"):
        return str(message.content)
    if isinstance(message, dict):
        for key in ("content", "text"):
            if key in message:
                return str(message[key])
    if isinstance(message, tuple) and len(message) >= 2:
        return str(message[1])
    return str(message)


class DummySearchAgent:
    def invoke(self, payload):
        query = payload.get("messages", [["", ""]])[-1][1]
        result = (
            "Dummy search mode is active. No live web search was performed.\n\n"
            f"Topic: {query}\n\n"
            "1. Official documentation or trusted source summary.\n"
            "2. Recent developments with verified references to key findings.\n"
            "3. Background context and practical implications.\n\n"
            "Source placeholders: https://example.com/source-1, https://example.com/source-2"
        )
        return {"messages": [{"content": result}]}


class DummyReaderAgent:
    def invoke(self, payload):
        prompt = payload.get("messages", [["", ""]])[-1][1]
        result = (
            "Dummy scraping mode is active. No remote page was fetched.\n\n"
            f"Context: {prompt[:500]}\n\n"
            "This placeholder result simulates a cleaned research excerpt from a trusted source."
        )
        return {"messages": [{"content": result}]}


def build_search_agent():
    if create_agent is not None and llm is not None:
        return create_agent(model=llm, tools=[web_search])
    return DummySearchAgent()


def build_reader_agent():
    if create_agent is not None and llm is not None:
        return create_agent(model=llm, tools=[scrape_url])
    return DummyReaderAgent()


def _dummy_report(topic, research):
    return (
        "# Research Report\n\n"
        f"## Introduction\nThis report covers {topic}. In dummy deployment mode, the system simulates a structured research workflow without external API access.\n\n"
        "## Key Findings\n"
        "1. The topic contains a credible research angle that benefits from multiple sources and careful synthesis.\n"
        "2. A high-quality report should separate confirmed facts from interpretation and highlight uncertainty where needed.\n"
        "3. The strongest conclusions come from combining source summaries, direct evidence, and a critical review step.\n\n"
        "## Conclusion\nThe dummy pipeline demonstrates the full research flow and validates the application logic, UI updates, and report generation path even without live model access.\n\n"
        "## Sources\n- https://example.com/source-1\n- https://example.com/source-2\n\n"
        f"### Raw research context\n{research[:2000]}"
    )


def _dummy_feedback(report):
    return (
        "Score: 8/10\n\n"
        "Strengths:\n"
        "- The structure is clear and easy to follow.\n"
        "- The report addresses the topic in a professional tone.\n\n"
        "Areas to Improve:\n"
        "- Add more source-specific evidence if live research data becomes available.\n"
        "- Include stronger comparative statements with direct citations.\n\n"
        "One line verdict: The dummy deployment confirms the workflow and report quality pipeline works as intended."
    )


class _WriterChain:
    def invoke(self, data):
        if llm is not None and ChatPromptTemplate is not None and StrOutputParser is not None:
            writer_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
                ("human", """Write a detailed report on the topic below.

                Topic: {topic}

                Research Gathered:
                {research}

                Structure the report as:
                - Introduction
                - Key Findings (minimum 3 well-explained points)
                - Conclusion
                - Sources (list all URLs found in the research)

                Be detailed, factual, and professional."""),
            ])
            return (writer_prompt | llm | StrOutputParser()).invoke(data)
        return _dummy_report(data.get("topic", "unknown topic"), data.get("research", ""))


class _CriticChain:
    def invoke(self, data):
        if llm is not None and ChatPromptTemplate is not None and StrOutputParser is not None:
            critic_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a sharp and constructive research critic. Be honest and specific."),
                ("human", """Review the research report below and evaluate it strictly.

                Report:
                {report}

                Respond in this exact format:

                Score: X/10

                Strengths:
                - ...
                - ...

                Areas to Improve:
                - ...
                - ...

                One line verdict:
                ..."""),
            ])
            return (critic_prompt | llm | StrOutputParser()).invoke(data)
        return _dummy_feedback(data.get("report", ""))


writer_chain = _WriterChain()
critic_chain = _CriticChain()
