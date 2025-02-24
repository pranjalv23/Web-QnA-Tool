from typing import Annotated, Sequence, Literal, TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages
from pydantic import BaseModel, Field

from core.logging_helpers import logger
from database.database_manager import create_retriever
from database.document_retriever import process_links
from core.backend import ingest_content


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


links = ingest_content()
docs_splits = process_links(urls=links)
retriever_tool = create_retriever(docs_splits)
tools = [retriever_tool]


def agent(state):
    logger.info("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def grade_documents(state) -> Literal["generate", "rewrite"]:
    logger.info("---CHECK RELEVANCE---")

    # Data model
    class Grade(BaseModel):

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4o", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(Grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        logger.info("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        logger.info("---DECISION: DOCS NOT RELEVANT---")
        logger.info(score)
        return "rewrite"


def rewrite(state):
    logger.info("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = ChatOpenAI(temperature=0.2, model="gpt-4o", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    logger.info("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.9, streaming=True)

    # Post-processing
    def format_docs(documents):
        return "\n\n".join(doc.page_content for doc in documents)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}
