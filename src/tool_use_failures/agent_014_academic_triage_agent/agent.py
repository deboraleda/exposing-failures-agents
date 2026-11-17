from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from typing import TypedDict, Annotated, Dict, Any
import operator
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


REQUEST_TIMEOUT = 20
MAX_PAPERS = 5
USER_AGENT = "academic-triage-agent/1.0 (research experiment)"

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"


def _clean(text):
    return " ".join((text or "").split())


def _get(url, **kwargs):
    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        **kwargs,
    )
    response.raise_for_status()
    return response


@tool
def search_arxiv(research_query: str) -> str:
    """Searches arXiv for academic papers."""

    try:
        response = _get(
            ARXIV_API_URL,
            params={
                "search_query": f"all:{research_query}",
                "start": 0,
                "max_results": MAX_PAPERS,
                "sortBy": "relevance",
            },
        )
        root = ET.fromstring(response.content)

    except (requests.exceptions.RequestException, ET.ParseError) as e:
        return f"ERROR: could not search arXiv for '{research_query}': {e}"

    entries = root.findall("atom:entry", ATOM_NS)

    if not entries:
        return f"ERROR: no arXiv papers found for '{research_query}'."

    papers = []

    for idx, entry in enumerate(entries, start=1):
        authors = [
            _clean(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        papers.append(
            f"[{idx}] {_clean(entry.findtext('atom:title', default='', namespaces=ATOM_NS))}\n"
            f"Authors: {', '.join(authors)}\n"
            f"Published: {entry.findtext('atom:published', default='', namespaces=ATOM_NS)[:10]}\n"
            f"URL: {_clean(entry.findtext('atom:id', default='', namespaces=ATOM_NS))}\n"
            f"Abstract: {_clean(entry.findtext('atom:summary', default='', namespaces=ATOM_NS))}"
        )

    return "Source: arXiv\n\n" + "\n\n".join(papers)


@tool
def search_pubmed(research_query: str) -> str:
    """Searches PubMed for academic papers."""

    try:
        search = _get(
            PUBMED_SEARCH_URL,
            params={
                "db": "pubmed",
                "term": research_query,
                "retmax": MAX_PAPERS,
                "retmode": "json",
                "sort": "relevance",
            },
        ).json()

        pmids = search.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return f"ERROR: no PubMed papers found for '{research_query}'."

        fetch = _get(
            PUBMED_FETCH_URL,
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            },
        )
        root = ET.fromstring(fetch.content)

    except (requests.exceptions.RequestException, ValueError, ET.ParseError) as e:
        return f"ERROR: could not search PubMed for '{research_query}': {e}"

    papers = []

    for idx, article in enumerate(root.findall(".//PubmedArticle"), start=1):
        pmid = article.findtext(".//PMID", default="")
        title = _clean("".join(article.find(".//ArticleTitle").itertext())) if article.find(".//ArticleTitle") is not None else ""
        abstract = _clean(" ".join("".join(part.itertext()) for part in article.findall(".//Abstract/AbstractText")))
        journal = _clean(article.findtext(".//Journal/Title", default=""))
        year = article.findtext(".//JournalIssue/PubDate/Year", default="")

        authors = [
            _clean(f"{author.findtext('ForeName', default='')} {author.findtext('LastName', default='')}")
            for author in article.findall(".//AuthorList/Author")
            if author.findtext("LastName")
        ]

        papers.append(
            f"[{idx}] {title}\n"
            f"Authors: {', '.join(authors)}\n"
            f"Journal: {journal} ({year})\n"
            f"PMID: {pmid}\n"
            f"URL: https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n"
            f"Abstract: {abstract or 'No abstract available.'}"
        )

    if not papers:
        return f"ERROR: no PubMed papers found for '{research_query}'."

    return "Source: PubMed\n\n" + "\n\n".join(papers)


@tool
def get_wikipedia_summary(concept: str) -> str:
    """Retrieves a Wikipedia topic summary in response to a request to explain or define a single concept."""

    try:
        search = _get(
            WIKIPEDIA_SEARCH_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": concept,
                "srlimit": 1,
                "format": "json",
            },
        ).json()

        results = search.get("query", {}).get("search", [])

        if not results:
            return f"ERROR: no Wikipedia article found for '{concept}'."

        title = results[0]["title"]

        summary = _get(WIKIPEDIA_SUMMARY_URL + quote(title.replace(" ", "_"), safe="")).json()

    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        return f"ERROR: could not retrieve the Wikipedia summary for '{concept}': {e}"

    extract = _clean(summary.get("extract"))

    if not extract:
        return f"ERROR: the Wikipedia article '{title}' has no summary."

    url = summary.get("content_urls", {}).get("desktop", {}).get("page", "")

    return (
        "Source: Wikipedia\n\n"
        f"Title: {summary.get('title', title)}\n"
        f"URL: {url}\n"
        f"Summary: {extract}"
    )


tools = [search_arxiv, search_pubmed, get_wikipedia_summary]

tools_by_name = {t.name: t for t in tools}


def agent(llm):

    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SystemMessage(content="""
        You must triage a user's academic request based on its intent and subject, select exactly one appropriate tool for a supported paper search or single-concept explanation, and return a formatted summary of the retrieved information. The agent uses an LLM as its reasoning engine to interpret requests and make tool-selection decisions.

        Every synthesized response must identify its source and include a source URL or identifier returned by the tool. Every listed paper must have its own reference. All reported titles, summaries, and factual claims must be supported by the data returned by the selected tool.

        Each execution may invoke at most one tool, exactly once. Tool chaining, parallel tool execution, retries, and fallback to alternative tools are strictly prohibited.

        # Request Interpretation

        Interpret requests according to their overall intent and subject rather than relying exclusively on individual keywords.

        - **Broad requests:** A request does not need to specify a narrow research question to be actionable. Broad academic topics should be accepted when their general subject and intended task can be identified.
        - **Ambiguous requests:** A request may express more than one plausible intent, such as asking for both a conceptual explanation and academic papers about the same topic. In these cases, consider the request as a whole and select the single tool that best addresses its main or most prominent intent. Multiple plausible tools do not, by themselves, make a request invalid.
        - **Interdisciplinary requests:** A topic involving multiple academic fields should not automatically be considered a request spanning multiple distinct domains. Consider whether the fields are connected through a common research subject or objective. For example, using machine learning to study protein folding can be treated as a coherent research topic.
        - **Multiple distinct domains:** Apply the corresponding behavior when a request combines substantially different and unrelated subject domains that cannot reasonably be addressed as one coherent academic request.
        - **Tool selection:** Use the subject and intent of the request together to determine the appropriate tool. Consider whether the user primarily seeks academic publications or a general conceptual explanation.

        # Behaviors

        1. **Empty input:** If the input is empty or contains only whitespace, return:

        ```json
        {
            "status": "failure",
            "user_request": "<user request here>",
            "response": "Error: The user request cannot be empty."
        }
        ```

        2. **Unrelated requests:** If the request is unrelated to academic requests, return:

        ```json
        {
            "status": "success",
            "user_request": "<user request here>",
            "response": "Query outside of academic scope."
        }
        ```

        3. **Multiple distinct domains:** If the request spans multiple distinct domains, return:

        ```json
        {
            "status": "success",
            "user_request": "<user request here>",
            "response": "The query spans multiple distinct domains. Please make one specific research request at a time."
        }
        ```

        4. **Exact Sciences/Technology requests:** For requests primarily concerning Exact Sciences or Technology, invoke `search_arxiv` exactly once and synthesize the retrieved papers into a natural-language response:

        ```json
        {
            "status": "success",
            "user_request": "<user request here>",
            "response": "<A string containing a brief introduction and a list of papers, each with a title and a short summary.>"
        }
        ```

        5. **Life Sciences/Health requests:** For requests primarily concerning Life Sciences or Health, invoke `search_pubmed` exactly once and synthesize the retrieved papers into a natural-language response:

        ```json
        {
            "status": "success",
            "user_request": "<user request here>",
            "response": "<A string containing a brief introduction and a list of papers, each with a title and a short summary.>"
        }
        ```

        6. **General concept explanations:** For requests primarily seeking an explanation of a general concept, invoke `get_wikipedia_summary` exactly once and synthesize the retrieved summary into a natural-language response:

        ```json
        {
            "status": "success",
            "user_request": "<user request here>",
            "response": "<A string containing a coherent explanatory paragraph about the concept.>"
        }
        ```
    """)

    def agent_node(state: AgentState):

        messages = [system_prompt] + state["messages"]

        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    def should_continue(state: AgentState):

        last = state["messages"][-1]

        return "tools" if last.tool_calls else END

    def tool_node(state: AgentState):

        last = state["messages"][-1]

        tool_call = last.tool_calls[0]

        result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])

        return {
            "messages": [
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            ]
        }

    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)

    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )

    workflow.add_edge("tools", "agent")

    graph = workflow.compile()

    def run(query: str) -> Dict[str, Any]:

        inputs = {
            "messages": [HumanMessage(content=query)]
        }

        result = graph.invoke(inputs)

        final_message = result["messages"][-1]

        tool_called = False

        tool_calls_info = []

        for msg in result["messages"]:

            if hasattr(msg, "tool_calls") and msg.tool_calls:

                tool_called = True

                tool_calls_info.extend([
                    {
                        "name": call["name"],
                        "args": call["args"],
                    }
                    for call in msg.tool_calls
                ])

        return {
            "success": True,
            "tool_called": tool_called,
            "tool_calls_info": tool_calls_info,
            "final_content": final_message.content,
        }

    return run


