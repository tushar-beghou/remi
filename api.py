import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.embeddings import OpenAIEmbeddings
from typing import Annotated, Sequence, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START

from neo4j import GraphDatabase
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from collections import defaultdict


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

embeddings = OpenAIEmbeddings()

API_KEY = os.getenv("GOOGLE_API_KEY")


API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client()
# def serach(query):
# model_id = "gemini-2.0-flash"
model = "gemini-2.5-flash-preview-04-17"

google_search_tool = Tool(
    google_search = GoogleSearch()
)


prompt = """
You are a highly knowledgeable and objective assistant specializing in the global aesthetics pharmaceutical market. Your purpose is to provide accurate, up-to-date, and detailed answers to questions related to this industry, including market size, trends, competitive landscape, regulatory dynamics, key players, product categories (e.g., botulinum toxins, dermal fillers), and regional differences.
You will also be provided a time filter, PROVIDE YOUR ANSWER ONLY FROM SOURCES FROM THE TIME PERIOD. Use reliable industry terminology and maintain a professional tone. When applicable, highlight market drivers, challenges, innovations, and forecasted growth. If data or specifics are not available, state that clearly rather than speculating. Prioritize clarity, relevance, and factual accuracy in all your responses. 

PLEASE PROVIDE ANSWER IN TIMELINE VIEW WITH DATES.

Here are instructions for questions you might frequently encounter:

1. What are the latest updates for Sculptra’s competitors ?
The following brands have competing products:
Allergan Aesthetics
Evolus
Sisram
Merz Aesthetics
Crown/Revance
Teoxane
Dermata Therapeutics
Dexlevo Inc
VIVACY
Burgeon Biotechnology

When retrieving information on latest updates, look at the following aspects – M&A deals or partnerships in a new market or for a new product or delivery mechanism, New delivery mechanism introduced for the product, new product feature or application area introduced, any new clinical trial updates, participation or activity in conferences or symposium related to healthcare or aesthetics market
Collect information from any worldwide press release and other specific aesthetics news links 
When sharing the response categorize the updates by the following categories: 
•	Product updates which includes product feature expansion, new market expansion, new application areas expansion
•	Innovation which includes new delivery mechanism
•	Trials and Conferences – clinical trials and conference participation and activity updates 


2. What are the competitive threats to Sculptra ?
The following brands have competing products:
Allergan Aesthetics
Evolus
Sisram
Merz Aesthetics
Crown/Revance
Teoxane
Dermata Therapeutics
Dexlevo Inc
VIVACY
Burgeon Biotechnology

Identify key competitors of Sculptra who are in the same product category and have same application areas
Compare product features of all competitors with Sculptra and identify unique or new features  
Compare latest press releases of competitors to identify any competitor with large number of partnership, M&A deals or new product feature releases 


Here are some key links to track such information:
https://aestheticmedicinenews.com/
https://www.medscape.com/resource/aesthetic-medicine
https://modernaesthetics.com/news
https://www.healio.com/news/dermatology/aesthetics
https://www.medestheticsmag.com/news
https://www.theaestheticguide.com/aesthetic-dermatology/injectables
https://aestheticmed.co.uk/industry-news
https://aestheticsjournal.com/news-blog/
https://bcam.ac.uk/media/news.aspx

PLEASE PROVIDE ANSWER IN TIMELINE VIEW WITH DATES.


TIME FILTER (ANSWER WITH SOURCES FROM):
"""
prompt_2 = """
Answer the following question:
"""


def opensearch(time_period,query):
    response = client.models.generate_content(
    model=model,
    contents=prompt + time_period + prompt_2 + query,
    config=GenerateContentConfig(
        tools=[google_search_tool],
        response_modalities=["TEXT"],
    )  
    )
    answer = ""
    references = set()

    answer = ""
    references_titles = []

    if response and response.candidates:
        first_candidate = response.candidates[0]
        if first_candidate.content and first_candidate.content.parts:
            for part in first_candidate.content.parts:
                if part.text:
                    answer += part.text

        if first_candidate.grounding_metadata and first_candidate.grounding_metadata.grounding_chunks:
            for chunk in first_candidate.grounding_metadata.grounding_chunks:
                if chunk.web and chunk.web.title:
                    references_titles.append(chunk.web.title)

        else:
            answer = "Apologies, I cannot help you with that"
    else:
        answer = "Apologies, I cannot help you with that"

    return answer, references_titles

entities = ['sculptra','radiesse','juvederm','botox','restylane','revance','zeomin','biostimulatory derma fillers']

driver = GraphDatabase.driver("neo4j+s://e2ac7cf9.databases.neo4j.io", auth=("neo4j", "kLgFm5MeLOz0erw8lck8FjqpU4ZggQVBEIBr2PvdoGU"))

def get_node_relationships_new(names):
    query = """
    UNWIND $names AS name
    MATCH (n {name: name})-[r]-(m)
    RETURN n.name AS node, type(r) AS rel_type, m.name AS connected_node
    ORDER BY node
    """
    with driver.session() as session:
        result = session.run(query, names=names)
        records = result.data()

    # Group relationships by node and then relationship type
    grouped = defaultdict(lambda: defaultdict(set))
    for record in records:
        node = record['node']
        rel_type = record['rel_type']
        connected_node = record['connected_node']
        grouped[node][rel_type].add(connected_node)

    # Format with <br><br> for HTML rendering in Streamlit
    output_parts = []
    for node, rels in grouped.items():
        parts = [f"<b>{node}</b>"]
        for rel_type, connected_nodes in rels.items():
            sorted_nodes = sorted(connected_nodes)
            connected_str = ", ".join(sorted_nodes)
            parts.append(f"<b>{rel_type}:</b> {connected_str}")
        output_parts.append("<br><br>".join(parts))

    return "<br><br>".join(output_parts)


def find_matching_words(input_string, word_list = entities):
    """
    Returns a list of words from word_list that are found in input_string.

    Args:
        input_string (str): The string to search within.
        word_list (list): List of words to look for in the input_string.

    Returns:
        list: Words from word_list that are found in input_string.
    """
    matches = [word for word in word_list if word in input_string]
    return matches

def rephrasal(query,context,history):
# Setup

        # context = get_node_relationships_new(final_query)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt_template = PromptTemplate.from_template("""
    Given the question history, follow-up question and additional context, rephrase the follow up question to be a standalone question.
    By rephrasing the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
                                                       

    Chat History:
    {history}
                                                   
    Question:
    {question}
                                                
    context:
    {context}                                           

    """)
    answer_grader = prompt_template | llm | StrOutputParser()

    ans = answer_grader.invoke({'history':history,'question': query,'context':context})
    return ans 



def make_tool_and_workflow():

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        keywords: list[str]
        context: str
        filters: str
        history: str
        rephrased_question: Annotated[Sequence[BaseMessage], add_messages]
        references: list[str]

    def rephrasal_first(state):
        messages = state['messages']
        question = messages[0].content
        history = state['history']
        filters = state['filters']

        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        prompt_template = PromptTemplate.from_template("""
        Given the question history and the follow-up question and a time period filter, rephrase the follow up question to be a standalone question.
        By rephrasing the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.
        
        Chat History:
        {history}
                                                    
        Question:
        {question}
                                                    
        Time Period:
        {time_period}
                                                                                            

        """)
        answer_grader = prompt_template | llm | StrOutputParser()

        ans = answer_grader.invoke({'history':history,'question': question,'time_period': filters})
        return {"messages":[ans], "rephrased_question": ans}
    
    def query_processor(state):

        messages = state['messages']
        query = state["rephrased_question"][-1]

        entities = find_matching_words(query)

        context = get_node_relationships_new(entities)

        rephrased_query_final = rephrasal(query, context, state['history'])

        answer, references = opensearch(state['filters'], rephrased_query_final)

        return {"messages":[answer], "context": context, "references": references, "keywords": entities}
    
    workflow = StateGraph(AgentState) 

    workflow.add_node("rephrasal_first", rephrasal_first)
    workflow.add_node("query_processor", query_processor)

    workflow.add_edge(START, "rephrasal_first")
    workflow.add_edge("rephrasal_first", "query_processor")
    workflow.add_edge("query_processor", END)

    graph = workflow.compile()

    return graph







    
