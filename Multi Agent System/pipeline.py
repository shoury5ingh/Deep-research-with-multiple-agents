from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


def _extract_message_text(message):
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


def run_research_pipeline(topic) -> dict:

    state = {}

    print("\n" + "="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })

    state["search_results"] = _extract_message_text(search_result['messages'][-1])

    print("\n search result ", state['search_results'])

    print("\n" + "="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })

    state['scraped_content'] = _extract_message_text(reader_result['messages'][-1])

    print("\nscraped content\n", state['scraped_content'])

    print("\n" + "="*50)
    print("step 3 - Writer is drafting a report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']}\n\n"
        f"Detailed Scraped Content : \n {state['scraped_content']}"
    )

    report = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    state["report"] = str(report)

    print("\n Final REport\n", state['report'])

    print("\n" + "="*50)
    print("step 4 - Critic is reviewing the report ...")
    print("="*50)

    feedback = critic_chain.invoke({
        "report": state['report']
    })
    state['feedback'] = str(feedback)

    print("\n Critic report\n", state['feedback'])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)

